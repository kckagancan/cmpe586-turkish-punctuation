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

```bash
python scripts/prepare_data.py
```

Bu komut `processed/clean-v1/` altında Hugging Face `dataset/`, ayrılan kayıtlar için
`quarantine/{train,validation,test}.jsonl` ve kaynak/ayar/kayıt ID'lerini içeren
`manifest.json` oluşturur. `clean-v1`, boş kelime içeren kaydın tamamını ayrı tutar;
kalan kelimeleri veya etiketleri değiştirmez. Çıktıların tamamı Git dışında kalır.
Hazırlanmış veri `datasets.load_from_disk("data/processed/clean-v1/dataset")`
ile açılabilir. Mevcut çıktı üzerine yazılmaz; politika değişirse yeni sürüm kullan.

## Noise varyantları

```bash
python scripts/prepare_noise.py
```

Yalnızca uygun validation kayıtlarından `processed/noise-v1/` oluşturur.
Mevcut clean-v1 çıktıları değişmez. Her koşul için:

- `conditions/<kosul>/records.jsonl`: aynı kayıt ID'leri ve etiketlerle gürültülü kelimeler.
- `edits.jsonl`: seçilen kelimeler, uygulanan işlem sırası, önce/sonra ve swap konumu.
- `dataset/`: yeniden tokenize edilmiş, ilk subword ile hizalanmış model girdileri.
- `audit.json`: gerçekleşen oranlar, örtüşmeler, sınıf bazında maruziyet ve tokenization.
- `examples.json`: seed ile seçilmiş küçük yerel örnekler.

Kök `manifest.json`, bütün koşulların raporlarını, ayar/kod/kaynak hash'lerini ve
çalışma bilgilerini içerir. `source_snapshot/` kullanılan kod/ayarları saklar;
`clean/dataset/` eski clean-v1 ile birebir karşılaştırılmış validation girdisidir.
Orijinal ayar dosyasının byte hash'i `source_snapshot/noise-config.json` kopyasına
aittir; kökteki `noise-config.json` aynı ayarların okunabilir biçimlendirilmiş kopyasıdır.
Örneğin bir varyant şöyle açılır:

```python
from datasets import load_from_disk

dataset = load_from_disk("data/processed/noise-v1/conditions/typo_swap-medium-seed586/dataset")
```

Model değerlendirmesi veya eğitim yapılmaz. Bütün bu yerel dosyalar Git dışındadır.
Gerekçeler: [noise protokolü](../docs/noise-protocol.md).
38 validation varyantının kontrol sonuçları: [noise veri raporu](../docs/noise-audit.md).
