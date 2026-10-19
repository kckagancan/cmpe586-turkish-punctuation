# Experiment configurations

Mevcut ayarlar JSON dosyalarında tutuluyor:

- `tokenizer.json`: yayımlanmış model/tokenizer kimliği ve sabit sürümü, aday temel
  model, önbellek, etiket eşlemesi ve kelime–subword hizalama kuralı.
- `preprocessing.json`: `clean-v1` hazırlama sürümü, boş kelimeli kayıt politikası,
  yerel çıktı klasörü, hazırlama/önizleme batch boyutları ve seed.
- `training.json`: `training-v1` fine-tuning protokolünün ayarları. Pinned BERTurk
  temel ağırlıkları, ortak başlangıç, optimizer/loss, effective batch, eğitim
  bütçesi, temiz validation ile checkpoint seçimi ve eğitimde noise karışımı
  tanımlıdır. Durumu `clean_pipeline_implemented_execution_pending`: clean training
  komutu ve Colab notebook'u yazıldı; henüz çalıştırılmadı. Train augmentation
  hazırlayıcısı plan aşamasında. Gerekçeler ve uygulama sırası:
  [eğitim protokolü](../docs/training-protocol.md).
- `evaluation.json`: ilk clean validation ölçümünün model girdisi/çıktı klasörleri,
  inference batch boyutu, cihaz/CPU thread ayarı, seed ve metrik tanımı. Ana skor,
  üç noktalama sınıfının eşit ağırlıklı macro-F1 değeridir; `O` konumları yanlış
  pozitifleri saymak için değerlendirmede korunur.
- `noise-evaluation.json`: kaydedilmiş noise koşullarının model değerlendirmesi.
  Temiz referans ve noise veri klasörlerini, seçilen koşulları, çıkarım ayarlarını,
  çıktı yolunu ve eşleştirilmiş hata örneklerinin sayısını belirler. İlk pilot:
  dört medium koşul, corruption seed 586. Buradaki `seed` çıkarım başlangıcı
  içindir; noise seçimi koşul adındaki seed ve hazırlama manifestiyle belirlenmiştir.
  `batch_order`, temiz token uzunluğuna göre tek bir sıralama kullanır; bütün
  koşullar aynı batch gruplarını paylaşır. `recompute_clean_anchor`, aynı sırayla
  temiz kontrol ölçümünü de üretir. Farklar bu eşleşen temiz ölçüme göre hesaplanır.
- `noise.json`: `noise-v1` dönüşümleri ve validation hazırlama ayarları. Tekil gürültü
  türlerinde tüm kelimelerin %2/%5/%10'u gerçekten değişir; seçimler seed ile
  tekrarlanır ve seviyeler aynı sıralamanın iç içe alt kümelerini kullanır.
  Türkçe harf kuralları, bir iç komşu harf takası, birleşik koşulun işlem sırası,
  tam harf kaybı koşulları ve veri kontrolleri tanımlıdır. `audit` bölümü tokenizer
  batch boyutunu ve her koşulda saklanan yerel örnek sayısını belirler. Gerekçeler
  ve sınırlar: [noise protokolü](../docs/noise-protocol.md).
  `python scripts/prepare_noise.py --config configs/noise.json` bu ayarlarla
  veriyi üretip kontrol eder; model ağırlığı yüklemez veya F1 ölçmez.

Yeni bir deneme için noise ayarını ayrı bir JSON'a kopyalayıp oran/seed ile
`output_directory` değerini değiştir. Mevcut çıktı üzerine yazılmaz. Aynı `version`
ve seed seçim sıralamasını korur; yeni oran aynı sıralamanın farklı uzunluktaki
ön ekini seçer. `version` değişirse hash anahtarı ve seçimler de değişir.
Kural değişikliğinde protokol/sürüm de güncellenir; yalnızca oran denemesinde
ayar hash'i ve yeni çıktı yolu denemeleri birbirinden ayırır.
Mevcut validation grid'inin veri kontrolü tamamlandı: [noise veri raporu](../docs/noise-audit.md).

Model ölçümü `python scripts/evaluate_noise.py` ile yapılır. Ayrı bir ayar dosyası
`--config` ile seçilir. Tamamlanmış sonuçların üzerine yazılmaz. Kesilmiş bir koşu
`--resume` ile yalnızca aynı ayar, kaynak kod ve girdi hash'leriyle sürdürülebilir;
tamamlanan koşullar korunur, yarım kalan koşul baştan hesaplanır. Temiz referansın
paket sürümleri ve çıkarım ayarları da karşılaştırmada eşleşmelidir.
Tamamlanan ilk pilotun sonuçları: [pilot raporu](../docs/noise-pilot.md).
`scripts/analyze_noise.py`, kaydedilmiş tahminlerden bağlam/değişen kelime hata
sayımını ve karşılaştırma tablosunu üretir; ek inference yapmaz.

Önizleme ve inference batch boyutları eğitim batch boyutu değildir. Eğitim için
başlangıç ayarı GPU başına 8 kayıt × 4 gradient accumulation = effective batch 32,
3 epoch ve learning rate 2e-5'tir. Bütün modellerde aynı başlangıç ağırlıkları,
kayıt sırası ve optimizer bütçesi kullanılacak. Augmented kolda her epoch kayıtların
yarısı combined-medium noisy input olarak sunulur; bu, kelimelerin %50'sinin bozulduğu
anlamına gelmez. Karışım için yeni ayarlar `training.json` içindedir; mevcut
`noise.json` dönüşümleri ve kaydedilmiş validation koşulları değiştirilmedi.

## Clean fine-tuned model evaluation

`clean-noise-evaluation.json` selects the verified local epoch export and the
published medium-noise pilot to match. It reuses existing clean/noisy validation
inputs. Run it with:

```bash
python scripts/evaluate_noise.py --config configs/clean-noise-evaluation.json
```

No training is performed. Protocol and results:
[clean-noise-pilot.md](../docs/clean-noise-pilot.md).
