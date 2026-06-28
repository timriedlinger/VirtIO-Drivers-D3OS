<div align="center">
  <a href="./Tim_Riedlinger_Bachelorarbeit.pdf">
    <img src="https://coconucos.cs.hhu.de/lehre/bigdata/resources/img/hhu-logo.svg" width="300">
  </a>

  <br><br>

  <a href="./Tim_Riedlinger_Bachelorarbeit.pdf">
    <img src="https://img.shields.io/static/v1?label=&message=pdf&color=EE3F24&style=for-the-badge&logo=adobe-acrobat-reader&logoColor=FFFFFF" alt="PDF">
  </a>
</div>

# :notebook: &nbsp; Aufgabenbeschreibung

# Ziele der Bachelorarbeit

- **VirtIO-Treiberbasis in D3OS integrieren**  
  - Virtio-GPU-Gerät erkennen und initialisieren  
  - Einbindung der Rust-Crate virtio-drivers als wiederverwendbare Treiberbasis 
  - Grundlage schaffen, um weitere VirtIO-Geräte später konsistent aktivieren zu können  

- **VirtIO-GPU Erweiterung**  
  - Resize Unterstützung für Auflösungsänderungen
  - VirGL Fähigkeit hinzufügen 

- **Demo-Anwendungen**  
  - VirGL Funktionstest 
  - VirtIO Sound Playback Test
  - Rectangle Demo anpassen auf Crate Implementierung

- **Performance-Test**  
  - Benchmarking der GPU-Integration
  - Vergleich WSL2 und Ubuntu25 als Host

# Build and Run

- **Ergänzung in Makefile.toml** 
  - "-display", "gtk,gl=on",
  - "-device", "virtio-vga-gl",
  - "-device", "virtio-sound-pci,audiodev=audio0",

- **Demo starten (in boot.rs)**
  - play_pcm_file();
  - test_virgl();
  - rectangle_demo(gpu_mutex);
  
- **Sonstiges** 
  - Git Version: 92cb6f3
  - Compiler: rustc 1.91.0-nightly (fe5536432 2025-08-29)
  - für VirGL wird virglrenderer Hostseitig vorausgesetzt
