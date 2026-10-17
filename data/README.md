# Local data

- `raw/`: indirilen orijinal dosyalar.
- `processed/`: işlenmiş temiz veri ve kontrollü noise türevleri.

Veri dosyaları Git'e eklenmez. Kaynak, lisans, indirme komutu, sürüm/hash ve
split bilgileri veri seçimi kesinleştiğinde dokümante edilecek.

Kurt & Çayır (2023) resmî deposundaki üç CSV, sabit bir commit üzerinden
`raw/kurt_cayir_2023/` içine indirildi. Kaynak linkleri ve SHA-256 değerleri
`docs/data-manifest.json` içinde. Veri lisansı henüz doğrulanmadı.

```bash
python scripts/download_data.py
python scripts/inspect_data.py
```

İlk inceleme: `docs/data-audit.md`. Ham dosyalara dönüşüm uygulanmadı.
Boş token'lar noktalama etiketi taşıyabiliyor; veri okuyucuda bunları sessizce silme.
