//! Driver for VirtIO GPU devices.

use crate::config::{read_config, ReadOnly, WriteOnly};
use crate::hal::{BufferDirection, Dma, Hal};
use crate::queue::VirtQueue;
use crate::transport::{InterruptStatus, Transport};
use crate::{pages, Error, Result, PAGE_SIZE};
use alloc::boxed::Box;
use bitflags::bitflags;
use log::{info, error};
use zerocopy::{FromBytes, FromZeros, Immutable, IntoBytes, KnownLayout};
use alloc::vec::Vec;

const QUEUE_SIZE: u16 = 16;
const SUPPORTED_FEATURES: Features = Features::RING_EVENT_IDX
    .union(Features::RING_INDIRECT_DESC)
    .union(Features::VERSION_1)
    .union(Features::VIRGL);

const SURFACE_ID_COLOR: u32 = 0x2001;

// VirGL virgl_object_type (from virgl_protocol.h)
const VIRGL_OBJECT_NULL: u32 = 0;
const VIRGL_OBJECT_RASTERIZER: u32 = 2;
const VIRGL_OBJECT_VERTEX_ELEMENTS: u32 = 5;
const VIRGL_OBJECT_SHADER: u32 = 4;
const VIRGL_OBJECT_SURFACE: u32 = 8;


// context cmds to be encoded in the command stream
const VIRGL_CCMD_CREATE_OBJECT: u32         = 1;
const VIRGL_CCMD_BIND_OBJECT: u32 = 2;
const VIRGL_CCMD_SET_VIEWPORT_STATE: u32    = 4;
const VIRGL_CCMD_SET_FRAMEBUFFER_STATE: u32 = 5;
const VIRGL_CCMD_SET_VERTEX_BUFFERS: u32    = 6;
const VIRGL_CCMD_CLEAR: u32                 = 7;
const VIRGL_CCMD_DRAW_VBO: u32              = 8;
const VIRGL_CCMD_RESOURCE_INLINE_WRITE: u32 = 9;
const VIRGL_CCMD_SET_SAMPLER_VIEWS: u32 = 10;
const VIRGL_CCMD_SET_INDEX_BUFFER: u32 = 11;
const VIRGL_CCMD_SET_CONSTANT_BUFFER: u32 = 12;
const VIRGL_CCMD_SET_SCISSOR_STATE: u32     = 15;
const VIRGL_CCMD_BIND_SHADER: u32           = 31;
const VIRGL_CCMD_SEND_STRING_MARKER: u32 = 51;
const VIRGL_CCMD_LINK_SHADER: u32 = 52;


#[inline]
fn virgl_cmd0(cmd: u32, obj: u32, len_u32: u32) -> u32 {
    // entspricht VIRGL_CMD0(cmd,obj,len)
    cmd | (obj << 8) | (len_u32 << 16)
}

#[inline]
fn push_u32(buf: &mut Vec<u8>, v: u32) { buf.extend_from_slice(&v.to_le_bytes()); }

#[inline]
fn push_f32(buf: &mut Vec<u8>, f: f32) { push_u32(buf, f.to_bits()); }

#[inline]
fn push_f64(buf: &mut Vec<u8>, d: f64) {
    let bits = d.to_bits();
    buf.extend_from_slice(&(bits as u32).to_le_bytes());         // low  32
    buf.extend_from_slice(&((bits >> 32) as u32).to_le_bytes()); // high 32
}

fn decode_hdr(h: u32) -> (u32,u32,u32) {
    let cmd =  h        & 0xff;
    let obj = (h >> 8)  & 0xff;
    let len =  h >> 16;
    (cmd, obj, len)
}

fn dump_cmds(cmds: &[u8]) {
    use core::convert::TryInto;
    let mut i = 0;
    while i + 4 <= cmds.len() {
        let h = u32::from_le_bytes(cmds[i..i+4].try_into().unwrap());
        let (cmd,obj,len) = decode_hdr(h);
        log::info!("hdr @{:03}: cmd={} obj={} len={}", i, cmd, obj, len);
        i += 4; // header
        let payload_bytes = (len as usize) * 4;
        // log::info!("  payload {} bytes", payload_bytes);
        i += payload_bytes;
    }
}

/// A virtio based graphics adapter.
///
/// It can operate in 2D mode and in 3D (virgl) mode.
/// 3D mode will offload rendering ops to the host gpu and therefore requires
/// a gpu with 3D support on the host machine.
/// In 2D mode the virtio-gpu device provides support for ARGB Hardware cursors
/// and multiple scanouts (aka heads).
pub struct VirtIOGpu<H: Hal, T: Transport> {
    transport: T,
    rect: Option<Rect>,
    /// DMA area of frame buffer.
    frame_buffer_dma: Option<Dma<H>>,
    /// DMA area of cursor image buffer.
    cursor_buffer_dma: Option<Dma<H>>,
    /// Queue for sending control commands.
    control_queue: VirtQueue<H, { QUEUE_SIZE as usize }>,
    /// Queue for sending cursor commands.
    cursor_queue: VirtQueue<H, { QUEUE_SIZE as usize }>,
    /// Send buffer for queue.
    queue_buf_send: Box<[u8]>,
    /// Recv buffer for queue.
    queue_buf_recv: Box<[u8]>,
}

impl<H: Hal, T: Transport> VirtIOGpu<H, T> {
    // Encoder für Commandstream - neu
    fn encode_create_surface(buf:&mut Vec<u8>, surf_handle:u32, res_handle: u32, format: u32){
        // create surface aus virgl_protocol.h
        push_u32(buf, virgl_cmd0(VIRGL_CCMD_CREATE_OBJECT, VIRGL_OBJECT_SURFACE, 5));
        push_u32(buf, surf_handle);
        push_u32(buf, res_handle);
        push_u32(buf, format);
        push_u32(buf, 0);   // level = 0
        push_u32(buf, 0);   // layers = 0 (first=0,last=0)
    }

    fn encode_set_framebuffer_state(buf:&mut Vec<u8>, color_surf:u32, depth_surf:u32){
        // framebuffer state aus virgl_protocol.h
        push_u32(buf, virgl_cmd0(VIRGL_CCMD_SET_FRAMEBUFFER_STATE, 0, 3));
        push_u32(buf, 1);            // nr_cbufs
        push_u32(buf, depth_surf);   // 0 = kein Depth
        push_u32(buf, color_surf);   // cbuf[0] = surface handle
    }

    fn encode_set_viewport_full(buf:&mut Vec<u8>, w:u32, h:u32){
        // viewport state aus virgl_protocol.h
        push_u32(buf, virgl_cmd0(VIRGL_CCMD_SET_VIEWPORT_STATE, 0, 7));

        // Payload
        push_u32(buf, 0);                  // VIRGL_SET_VIEWPORT_START_SLOT
        push_f32(buf, (w as f32) * 0.5);   // SCALE_0
        push_f32(buf, (h as f32) * 0.5);   // SCALE_1
        push_f32(buf, 1.0);                // SCALE_2
        push_f32(buf, (w as f32) * 0.5);   // TRANSLATE_0
        push_f32(buf, (h as f32) * 0.5);   // TRANSLATE_1
        push_f32(buf, 0.0);                // TRANSLATE_2
    }

    fn encode_clear_color(buf:&mut Vec<u8>, r:f32,g:f32,b:f32,a:f32){
        // #define VIRGL_OBJ_CLEAR_SIZE 8 aus virgl_protocol.h
        push_u32(buf, virgl_cmd0(VIRGL_CCMD_CLEAR, 0, 8));

        const PIPE_CLEAR_COLOR0: u32 = 1 << 2; // übliches Gallium-Flag für Color0
        push_u32(buf, PIPE_CLEAR_COLOR0);
                 
        push_f32(buf, r);
        push_f32(buf, g);
        push_f32(buf, b);
        push_f32(buf, a);

        push_f64(buf, 1.0f64); // depth (irrelevant bei farb-clear)
        push_u32(buf, 0); // stencil
    }

    
    /// Create a new VirtIO-Gpu driver.
    pub fn new(mut transport: T) -> Result<Self> {
        let negotiated_features = transport.begin_init(SUPPORTED_FEATURES);
        info!(
            "[gpu device] negotiated_features: {:?}",
            negotiated_features
        );

        // read configuration space
        let events_read = read_config!(transport, Config, events_read)?;
        let num_scanouts = read_config!(transport, Config, num_scanouts)?;
        info!(
            "events_read: {:#x}, num_scanouts: {:#x}",
            events_read, num_scanouts
        );

        let control_queue = VirtQueue::new(
            &mut transport,
            QUEUE_TRANSMIT,
            negotiated_features.contains(Features::RING_INDIRECT_DESC),
            negotiated_features.contains(Features::RING_EVENT_IDX),
        )?;
        let cursor_queue = VirtQueue::new(
            &mut transport,
            QUEUE_CURSOR,
            negotiated_features.contains(Features::RING_INDIRECT_DESC),
            negotiated_features.contains(Features::RING_EVENT_IDX),
        )?;

        let queue_buf_send = FromZeros::new_box_zeroed_with_elems(PAGE_SIZE).unwrap();
        let queue_buf_recv = FromZeros::new_box_zeroed_with_elems(PAGE_SIZE).unwrap();

        transport.finish_init();

        Ok(VirtIOGpu {
            transport,
            frame_buffer_dma: None,
            cursor_buffer_dma: None,
            rect: None,
            control_queue,
            cursor_queue,
            queue_buf_send,
            queue_buf_recv,
        })
    }

    /// Acknowledge interrupt.
    pub fn ack_interrupt(&mut self) -> InterruptStatus {
        self.transport.ack_interrupt()
    }

    /// Get the resolution (width, height).
    pub fn resolution(&mut self) -> Result<(u32, u32)> {
        let display_info = self.get_display_info()?;
        Ok((display_info.rect.width, display_info.rect.height))
    }

    /// Setup framebuffer - angepasst
    pub fn setup_framebuffer(&mut self) -> Result<&mut [u8]> {
        // get display info
        let display_info = self.get_display_info()?;
        info!("=> {:?}", display_info);

        // Wenn schon ein FB existiert: erst killen
        if self.frame_buffer_dma.is_some() {
            // nimm alten rect, sonst den aktuellen
            let r = self.rect.unwrap_or(display_info.rect);
            self.destroy_framebuffer(r);
        }
        
        self.rect = Some(display_info.rect);

        // create resource 2d
        self.resource_create_2d(
            RESOURCE_ID_FB,
            display_info.rect.width,
            display_info.rect.height,
        )?;

        // alloc continuous pages for the frame buffer
        let size = display_info.rect.width * display_info.rect.height * 4;
        let frame_buffer_dma = Dma::new(pages(size as usize), BufferDirection::DriverToDevice)?;

        // resource_attach_backing
        self.resource_attach_backing(RESOURCE_ID_FB, frame_buffer_dma.paddr() as u64, size)?;

        // map frame buffer to screen
        self.set_scanout(display_info.rect, SCANOUT_ID, RESOURCE_ID_FB)?;

        // SAFETY: `Dma::new` guarantees that the pointer returned from
        // `raw_slice` is non-null, aligned, and the allocation is zeroed. We
        // store the `Dma` object in `self.frame_buffer_dma`, which prevents the
        // allocation from being freed while `self` exists. The returned ptr
        // borrows `self` mutably, which prevents other code from getting
        // another reference to `frame_buffer_dma` while the returned slice is
        // still in use.
        let buf = unsafe { frame_buffer_dma.raw_slice().as_mut() };
        self.frame_buffer_dma = Some(frame_buffer_dma);
        Ok(buf)
    }
    // neu - für resize
    fn destroy_framebuffer(&mut self, rect_for_scanout: Rect) {
        // Scanout lösen (Host soll nicht mehr aus alter Resource scannen)
        let _ = self.set_scanout(rect_for_scanout, SCANOUT_ID, 0);

        // Backing lösen + 3) Resource freigeben
        let _ = self.resource_detach_backing(RESOURCE_ID_FB);
        let _ = self.resource_unref(RESOURCE_ID_FB);

        // Treiber-State droppen (damit DMA freigegeben wird)
        self.frame_buffer_dma = None;
        self.rect = None;
    }

    /// Flush framebuffer to screen.
    pub fn flush(&mut self) -> Result {
        let rect = self.rect.ok_or(Error::NotReady)?;
        // copy data from guest to host
        self.transfer_to_host_2d(rect, 0, RESOURCE_ID_FB)?;
        // flush data to screen
        self.resource_flush(rect, RESOURCE_ID_FB)?;
        Ok(())
    }

    /// Set the pointer shape and position.
    pub fn setup_cursor(
        &mut self,
        cursor_image: &[u8],
        pos_x: u32,
        pos_y: u32,
        hot_x: u32,
        hot_y: u32,
    ) -> Result {
        let size = CURSOR_RECT.width * CURSOR_RECT.height * 4;
        if cursor_image.len() != size as usize {
            return Err(Error::InvalidParam);
        }
        let cursor_buffer_dma = Dma::new(pages(size as usize), BufferDirection::DriverToDevice)?;

        // SAFETY: `Dma::new` guarantees that the pointer returned from
        // `raw_slice` is non-null, aligned, and the allocation is zeroed. The
        // returned reference is only used within this function while
        // `cursor_buffer_dma` is alive.
        let buf = unsafe { cursor_buffer_dma.raw_slice().as_mut() };
        buf.copy_from_slice(cursor_image);

        self.resource_create_2d(RESOURCE_ID_CURSOR, CURSOR_RECT.width, CURSOR_RECT.height)?;
        self.resource_attach_backing(RESOURCE_ID_CURSOR, cursor_buffer_dma.paddr() as u64, size)?;
        self.transfer_to_host_2d(CURSOR_RECT, 0, RESOURCE_ID_CURSOR)?;
        self.update_cursor(
            RESOURCE_ID_CURSOR,
            SCANOUT_ID,
            pos_x,
            pos_y,
            hot_x,
            hot_y,
            false,
        )?;
        self.cursor_buffer_dma = Some(cursor_buffer_dma);
        Ok(())
    }

    /// Move the pointer without updating the shape.
    pub fn move_cursor(&mut self, pos_x: u32, pos_y: u32) -> Result {
        self.update_cursor(RESOURCE_ID_CURSOR, SCANOUT_ID, pos_x, pos_y, 0, 0, true)?;
        Ok(())
    }

    /// Send a request to the device and block for a response.
    fn request<Req: IntoBytes + Immutable, Rsp: FromBytes>(&mut self, req: Req) -> Result<Rsp> {
        req.write_to_prefix(&mut self.queue_buf_send).unwrap();
        self.control_queue.add_notify_wait_pop(
            &[&self.queue_buf_send],
            &mut [&mut self.queue_buf_recv],
            &mut self.transport,
        )?;
        Ok(Rsp::read_from_prefix(&self.queue_buf_recv).unwrap().0)
    }

    /// Sendet eine Anfrage mit einem zusätzlichen Datenpuffer an das Gerät.
    fn request_with_data<Req: IntoBytes + Immutable, Rsp: FromBytes>(&mut self,req: Req, data: &[u8]) -> Result<Rsp> {
        // Schreibe den Header in den Sendepuffer.
        req.write_to_prefix(&mut self.queue_buf_send).unwrap();
        let header_len = core::mem::size_of::<Req>();

        // Sende Header und Daten als zwei getrennte Puffer in einer Kette.
        self.control_queue.add_notify_wait_pop(
            &[&self.queue_buf_send[..header_len], data],
            &mut [&mut self.queue_buf_recv],
            &mut self.transport,
        )?;
        Ok(Rsp::read_from_prefix(&self.queue_buf_recv).unwrap().0)
    }

    /// Send a mouse cursor operation request to the device and block for a response.
    fn cursor_request<Req: IntoBytes + Immutable>(&mut self, req: Req) -> Result {
        req.write_to_prefix(&mut self.queue_buf_send).unwrap();
        self.cursor_queue.add_notify_wait_pop(
            &[&self.queue_buf_send],
            &mut [],
            &mut self.transport,
        )?;
        Ok(())
    }

    fn get_display_info(&mut self) -> Result<RespDisplayInfo> {
        let info: RespDisplayInfo =
            self.request(CtrlHeader::with_type(Command::GET_DISPLAY_INFO))?;
        info.header.check_type(Command::OK_DISPLAY_INFO)?;
        Ok(info)
    }

    fn resource_create_2d(&mut self, resource_id: u32, width: u32, height: u32) -> Result {
        let rsp: CtrlHeader = self.request(ResourceCreate2D {
            header: CtrlHeader::with_type(Command::RESOURCE_CREATE_2D),
            resource_id,
            format: Format::B8G8R8A8UNORM,
            width,
            height,
        })?;
        rsp.check_type(Command::OK_NODATA)
    }

    ///Erstellt 3D resource Create
    fn resource_create_3d(&mut self, resource_id: u32, target: u32, format: u32, bind: u32, width: u32, height: u32, depth: u32, array_size: u32, last_level: u32, nr_samples: u32, flags: u32) -> Result {
        let rsp: CtrlHeader = self.request(ResourceCreate3D {
            header: CtrlHeader::with_type(Command::RESOURCE_CREATE_3D),
            resource_id,
            target,
            format,
            bind,
            width,
            height,
            depth,
            array_size,
            last_level,
            nr_samples,
            flags,
            _padding: 0,
        })?;
        rsp.check_type(Command::OK_NODATA)
    }

    ///Erstellt context resource Create
    fn context_create(&mut self, ctx_id: u32, name: &str) -> Result {
        let mut header = CtrlHeader::with_type(Command::CTX_CREATE);
        header.ctx_id = ctx_id;

        let mut dbg = [0u8; 64];
        let bytes = name.as_bytes();
        let n = core::cmp::min(bytes.len(), dbg.len());
        dbg[..n].copy_from_slice(&bytes[..n]);

        // Erstelle die CtxCreate-Struktur ohne den variablen Teil.
        let create_cmd = CtxCreate {
            header,
            nlen: name.as_bytes().len() as u32,
            _padding: 0,
            debug_name: dbg,
        };
        
        // Sende den Header und den Namen als separaten Puffer.
        let rsp: CtrlHeader = self.request_with_data(create_cmd, name.as_bytes())?;
        rsp.check_type(Command::OK_NODATA)
    }

    ///Submit 3D
    fn submit_3d(&mut self, ctx_id: u32, commands: &[u8]) -> Result {
        let mut header = CtrlHeader::with_type(Command::CMD_SUBMIT_3D);
        header.ctx_id = ctx_id;

        static mut NEXT_FENCE: u64 = 1;
        let fence = unsafe { let f = NEXT_FENCE; NEXT_FENCE += 1; f };
        header.flags = GPU_FLAG_FENCE;
        header.fence_id = fence;

        // Erstelle die CmdSubmit3D-Struktur.
        let submit_cmd = CmdSubmit3D {
            header,
            size: commands.len() as u32,
            _padding: 0,
        };

        // Sende den Header und den 3D-Befehlspuffer.
        let rsp: CtrlHeader = self.request_with_data(submit_cmd, commands)?;
            info!("ctx_submit rsp type={:#x}, fence_id={:#x}, ctx={:#x}",
            rsp.hdr_type.0, rsp.fence_id, rsp.ctx_id);
        rsp.check_type(Command::OK_NODATA)
    }

    fn set_scanout(&mut self, rect: Rect, scanout_id: u32, resource_id: u32) -> Result {
        let rsp: CtrlHeader = self.request(SetScanout {
            header: CtrlHeader::with_type(Command::SET_SCANOUT),
            rect,
            scanout_id,
            resource_id,
        })?;
        rsp.check_type(Command::OK_NODATA)
    }

    fn resource_flush(&mut self, rect: Rect, resource_id: u32) -> Result {
        let rsp: CtrlHeader = self.request(ResourceFlush {
            header: CtrlHeader::with_type(Command::RESOURCE_FLUSH),
            rect,
            resource_id,
            _padding: 0,
        })?;
        rsp.check_type(Command::OK_NODATA)
    }

    fn transfer_to_host_2d(&mut self, rect: Rect, offset: u64, resource_id: u32) -> Result {
        let rsp: CtrlHeader = self.request(TransferToHost2D {
            header: CtrlHeader::with_type(Command::TRANSFER_TO_HOST_2D),
            rect,
            offset,
            resource_id,
            _padding: 0,
        })?;
        rsp.check_type(Command::OK_NODATA)
    }

    fn transfer_to_host_3d(&mut self, box_3d: Box3D, offset: u64, resource_id: u32, level: u32, stride: u32, layer_stride: u32) -> Result {
        let rsp: CtrlHeader = self.request(TransferToHost3D {
            header: CtrlHeader::with_type(Command::TRANSFER_TO_HOST_3D),
            box_3d,
            offset,
            resource_id,
            level,
            stride,
            layer_stride,
        })?;
        rsp.check_type(Command::OK_NODATA)
    }

    fn transfer_from_host_3d(&mut self, box_3d: Box3D, offset: u64, resource_id: u32, level: u32, stride: u32, layer_stride: u32) -> Result {
        let rsp: CtrlHeader = self.request(TransferFromHost3D {
            header: CtrlHeader::with_type(Command::TRANSFER_FROM_HOST_3D),
            box_3d,
            offset,
            resource_id,
            level,
            stride,
            layer_stride,
        })?;
        rsp.check_type(Command::OK_NODATA)
    }

    fn ctx_attach_resource(&mut self, ctx_id: u32, resource_id: u32) -> Result {
        let mut header = CtrlHeader::with_type(Command::CTX_ATTACH_RESOURCE);
        header.ctx_id = ctx_id;

        let rsp: CtrlHeader = self.request(CtxAttachResource {
            header,
            resource_id,
            _padding: 0,
        })?;
        rsp.check_type(Command::OK_NODATA)
    }

    fn resource_attach_backing(&mut self, resource_id: u32, paddr: u64, length: u32) -> Result {
        let rsp: CtrlHeader = self.request(ResourceAttachBacking {
            header: CtrlHeader::with_type(Command::RESOURCE_ATTACH_BACKING),
            resource_id,
            nr_entries: 1,
            addr: paddr,
            length,
            _padding: 0,
        })?;
        rsp.check_type(Command::OK_NODATA)
    }

    fn resource_detach_backing(&mut self, resource_id: u32) -> Result {
        let rsp: CtrlHeader = self.request(ResourceDetachBacking {
            header: CtrlHeader::with_type(Command::RESOURCE_DETACH_BACKING),
            resource_id,
            _padding: 0,
        })?;
        rsp.check_type(Command::OK_NODATA)
    }

    fn resource_unref(&mut self, resource_id: u32) -> Result {
        let rsp: CtrlHeader = self.request(ResourceUnref {
            header: CtrlHeader::with_type(Command::RESOURCE_UNREF),
            resource_id,
            _padding: 0,
        })?;
        rsp.check_type(Command::OK_NODATA)
    }

    fn update_cursor(
        &mut self,
        resource_id: u32,
        scanout_id: u32,
        pos_x: u32,
        pos_y: u32,
        hot_x: u32,
        hot_y: u32,
        is_move: bool,
    ) -> Result {
        self.cursor_request(UpdateCursor {
            header: if is_move {
                CtrlHeader::with_type(Command::MOVE_CURSOR)
            } else {
                CtrlHeader::with_type(Command::UPDATE_CURSOR)
            },
            pos: CursorPos {
                scanout_id,
                x: pos_x,
                y: pos_y,
                _padding: 0,
            },
            resource_id,
            hot_x,
            hot_y,
            _padding: 0,
        })
    }

    /// Führt einen Smoke-Test für die Virgl-3D-Funktionalität aus.
    pub fn test_virgl(&mut self) -> Result<()> {
        info!("\n Starting Virgl 3D test...");

        // 1) Context erstellen
        let ctx_id = 1;
        self.context_create(ctx_id, "test_ctx")?;

        // 2) Auflösung holen
        let (width, height) = self.resolution()?;
        let rect = Rect { x: 0, y: 0, width, height };

        // 3) 3D-Ressource erstellen
        let res_id = 0x2000;
        let bind_flags = 3;
        self.resource_create_3d(
            res_id, 
            2,          // target: PIPE_TEXTURE_2D
            1,          // format: B8G8R8A8_UNORM (VirtIO enum value)
            bind_flags, // bind: SCANOUT | RENDER_TARGET
            width, height, 1, 1, 0, 0, 0
        )?;

        // Ressource an Kontext binden
        self.ctx_attach_resource(ctx_id, res_id)?;

        // 5) Scanout direkt auf 3D-Ressource
        self.set_scanout(rect, SCANOUT_ID, res_id)?;

        // 6) Commandstream bauen (Grüner Hintergrund)
        let mut cmds = Vec::new();
        // Surface für res_id erstellen
        Self::encode_create_surface(&mut cmds, SURFACE_ID_COLOR, res_id, 1);

        // Framebuffer binden
        Self::encode_set_framebuffer_state(&mut cmds, SURFACE_ID_COLOR, VIRGL_OBJECT_NULL);

        // Viewport setzen
        Self::encode_set_viewport_full(&mut cmds, width, height);

        // // Clear Screen (RGBA)
        Self::encode_clear_color(&mut cmds, 0.094, 0.416, 0.067, 1.0);
        dump_cmds(&cmds);

        info!("Sende {} Bytes an 3D-Befehlen...", cmds.len());

        // 7) Submit an GPU
        self.submit_3d(ctx_id, &cmds)?;

        // 8) Flush (theoretisch auch ohne Flush möglich)
        self.resource_flush(rect, res_id)?;

        info!("Rendered green frame and displayed successfully!");
        Ok(())
    }
}

impl<H: Hal, T: Transport> Drop for VirtIOGpu<H, T> {
    fn drop(&mut self) {
        // Clear any pointers pointing to DMA regions, so the device doesn't try to access them
        // after they have been freed.
        self.transport.queue_unset(QUEUE_TRANSMIT);
        self.transport.queue_unset(QUEUE_CURSOR);
    }
}

#[repr(C)]
struct Config {
    /// Signals pending events to the driver。
    events_read: ReadOnly<u32>,

    /// Clears pending events in the device.
    events_clear: WriteOnly<u32>,

    /// Specifies the maximum number of scanouts supported by the device.
    ///
    /// Minimum value is 1, maximum value is 16.
    num_scanouts: ReadOnly<u32>,
}

/// Display configuration has changed.
const EVENT_DISPLAY: u32 = 1 << 0;

bitflags! {
    #[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
    struct Features: u64 {
        /// virgl 3D mode is supported.
        const VIRGL                 = 1 << 0;
        /// EDID is supported.
        const EDID                  = 1 << 1;

        // device independent
        const NOTIFY_ON_EMPTY       = 1 << 24; // legacy
        const ANY_LAYOUT            = 1 << 27; // legacy
        const RING_INDIRECT_DESC    = 1 << 28;
        const RING_EVENT_IDX        = 1 << 29;
        const UNUSED                = 1 << 30; // legacy
        const VERSION_1             = 1 << 32; // detect legacy

        // since virtio v1.1
        const ACCESS_PLATFORM       = 1 << 33;
        const RING_PACKED           = 1 << 34;
        const IN_ORDER              = 1 << 35;
        const ORDER_PLATFORM        = 1 << 36;
        const SR_IOV                = 1 << 37;
        const NOTIFICATION_DATA     = 1 << 38;
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, FromBytes, Immutable, IntoBytes, KnownLayout, PartialEq)]
struct Command(u32);

impl Command {
    const GET_DISPLAY_INFO: Command = Command(0x100);
    const RESOURCE_CREATE_2D: Command = Command(0x101);
    const RESOURCE_UNREF: Command = Command(0x102);
    const SET_SCANOUT: Command = Command(0x103);
    const RESOURCE_FLUSH: Command = Command(0x104);
    const TRANSFER_TO_HOST_2D: Command = Command(0x105);
    const RESOURCE_ATTACH_BACKING: Command = Command(0x106);
    const RESOURCE_DETACH_BACKING: Command = Command(0x107);
    const GET_CAPSET_INFO: Command = Command(0x108);
    const GET_CAPSET: Command = Command(0x109);
    const GET_EDID: Command = Command(0x10a);

    // 3D Commands - neu
    const CTX_CREATE: Command = Command(0x200);
    const CTX_DESTROY: Command = Command(0x201);
    const CTX_ATTACH_RESOURCE: Command = Command(0x202);
    const CTX_DETACH_RESOURCE: Command = Command(0x203);
    const RESOURCE_CREATE_3D: Command = Command(0x204);
    const TRANSFER_TO_HOST_3D: Command = Command(0x205);
    const TRANSFER_FROM_HOST_3D: Command = Command(0x206);
    const CMD_SUBMIT_3D: Command = Command(0x207);

    const UPDATE_CURSOR: Command = Command(0x300);
    const MOVE_CURSOR: Command = Command(0x301);

    const OK_NODATA: Command = Command(0x1100);
    const OK_DISPLAY_INFO: Command = Command(0x1101);
    const OK_CAPSET_INFO: Command = Command(0x1102);
    const OK_CAPSET: Command = Command(0x1103);
    const OK_EDID: Command = Command(0x1104);

    const ERR_UNSPEC: Command = Command(0x1200);
    const ERR_OUT_OF_MEMORY: Command = Command(0x1201);
    const ERR_INVALID_SCANOUT_ID: Command = Command(0x1202);
}

const GPU_FLAG_FENCE: u32 = 1 << 0;

#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, Immutable, IntoBytes, KnownLayout)]
struct CtrlHeader {
    hdr_type: Command,
    flags: u32,
    fence_id: u64,
    ctx_id: u32,
    _padding: u32,
}

impl CtrlHeader {
    fn with_type(hdr_type: Command) -> CtrlHeader {
        CtrlHeader {
            hdr_type,
            flags: 0,
            fence_id: 0,
            ctx_id: 0,
            _padding: 0,
        }
    }

    /// Return error if the type is not same as expected.
    fn check_type(&self, expected: Command) -> Result {
        if self.hdr_type == expected {
            Ok(())
        } else {
            Err(Error::IoError)
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default, FromBytes, Immutable, IntoBytes, KnownLayout)]
struct Rect {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

#[repr(C)]
#[derive(Debug, FromBytes, Immutable, KnownLayout)]
struct RespDisplayInfo {
    header: CtrlHeader,
    rect: Rect,
    enabled: u32,
    flags: u32,
}

#[repr(C)]
#[derive(Debug, Immutable, IntoBytes, KnownLayout)]
struct ResourceCreate2D {
    header: CtrlHeader,
    resource_id: u32,
    format: Format,
    width: u32,
    height: u32,
}

#[repr(u32)]
#[derive(Debug, Immutable, IntoBytes, KnownLayout)]
enum Format {
    B8G8R8A8UNORM = 1,
}

#[repr(C)]
#[derive(Debug, Immutable, IntoBytes, KnownLayout)]
struct ResourceAttachBacking {
    header: CtrlHeader,
    resource_id: u32,
    nr_entries: u32, // always 1
    addr: u64,
    length: u32,
    _padding: u32,
}
// neu
#[repr(C)]
#[derive(Debug, Immutable, IntoBytes, KnownLayout)]
struct ResourceDetachBacking {
    header: CtrlHeader,
    resource_id: u32,
    _padding: u32,
}

// neu
#[repr(C)]
#[derive(Debug, Immutable, IntoBytes, KnownLayout)]
struct ResourceUnref {
    header: CtrlHeader,
    resource_id: u32,
    _padding: u32,
}


#[repr(C)]
#[derive(Debug, Immutable, IntoBytes, KnownLayout)]
struct SetScanout {
    header: CtrlHeader,
    rect: Rect,
    scanout_id: u32,
    resource_id: u32,
}

#[repr(C)]
#[derive(Debug, Immutable, IntoBytes, KnownLayout)]
struct TransferToHost2D {
    header: CtrlHeader,
    rect: Rect,
    offset: u64,
    resource_id: u32,
    _padding: u32,
}

#[repr(C)]
#[derive(Debug, Immutable, IntoBytes, KnownLayout)]
struct ResourceFlush {
    header: CtrlHeader,
    rect: Rect,
    resource_id: u32,
    _padding: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Immutable, IntoBytes, KnownLayout)]
struct CursorPos {
    scanout_id: u32,
    x: u32,
    y: u32,
    _padding: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Immutable, IntoBytes, KnownLayout)]
struct UpdateCursor {
    header: CtrlHeader,
    pos: CursorPos,
    resource_id: u32,
    hot_x: u32,
    hot_y: u32,
    _padding: u32,
}

//3D related - neu

#[repr(C)]
struct Vertex {
    pos: [f32; 4],
    // color: [f32; 3],
}


#[repr(C)]
#[derive(Debug, Immutable, IntoBytes, KnownLayout)]
struct ResourceCreate3D {
    header: CtrlHeader,
    resource_id: u32,
    target: u32,
    format: u32,
    bind: u32,
    width: u32,
    height: u32,
    depth: u32,
    array_size: u32,
    last_level: u32,
    nr_samples: u32,
    flags: u32,
    _padding: u32,
}

#[repr(C)]
#[derive(Debug, Immutable, IntoBytes, KnownLayout)]
struct CtxCreate {
    header: CtrlHeader,
    nlen: u32,
    _padding: u32,
    debug_name: [u8; 64],
}

#[repr(C)]
#[derive(Debug, Immutable, IntoBytes, KnownLayout)]
struct CtxDestroy {
    header: CtrlHeader,
}

#[repr(C)]
#[derive(Debug, Immutable, IntoBytes, KnownLayout)]
struct CtxAttachResource {
    header: CtrlHeader,
    resource_id: u32,
    _padding: u32,
}

#[repr(C)]
#[derive(Debug, Immutable, IntoBytes, KnownLayout)]
struct CtxDetachResource {
    header: CtrlHeader,
    resource_id: u32,
    _padding: u32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default, FromBytes, Immutable, IntoBytes, KnownLayout)]
struct Box3D {
    x: u32,
    y: u32,
    z: u32,
    w: u32,
    h: u32,
    d: u32,
}

#[repr(C)]
#[derive(Debug, Immutable, IntoBytes, KnownLayout)]
struct TransferToHost3D {
    header: CtrlHeader,
    box_3d: Box3D,
    offset: u64,
    resource_id: u32,
    level: u32,
    stride: u32,
    layer_stride: u32,
}

#[repr(C)]
#[derive(Debug, Immutable, IntoBytes, KnownLayout)]
struct TransferFromHost3D {
    header: CtrlHeader,
    box_3d: Box3D,
    offset: u64,
    resource_id: u32,
    level: u32,
    stride: u32,
    layer_stride: u32,
}

#[repr(C)]
#[derive(Debug, Immutable, IntoBytes, KnownLayout)]
struct CmdSubmit3D {
    header: CtrlHeader,
    size: u32,
    _padding: u32,
    // Befehlspuffer als Byte-Slice
}

const QUEUE_TRANSMIT: u16 = 0;
const QUEUE_CURSOR: u16 = 1;

const SCANOUT_ID: u32 = 0;
const RESOURCE_ID_FB: u32 = 0xbabe;
const RESOURCE_ID_CURSOR: u32 = 0xdade;

const CURSOR_RECT: Rect = Rect {
    x: 0,
    y: 0,
    width: 64,
    height: 64,
};
