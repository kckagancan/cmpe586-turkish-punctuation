# Experiment configurations

Mevcut ayarlar JSON dosyalarında tutuluyor:

- `tokenizer.json`: yayımlanmış model/tokenizer kimliği ve sabit sürümü, aday temel
  model, önbellek, etiket eşlemesi ve kelime–subword hizalama kuralı.
- `preprocessing.json`: `clean-v1` hazırlama sürümü, boş kelimeli kayıt politikası,
  yerel çıktı klasörü, hazırlama/önizleme batch boyutları ve seed.
- `training.json`: `training-v1` fine-tuning protokolünün ayarları. Pinned BERTurk
  temel ağırlıkları, ortak başlangıç, optimizer/loss, effective batch, eğitim
  bütçesi, temiz validation ile checkpoint seçimi ve eğitimde noise karışımı
  tanımlıdır. Clean training üç epoch ile tamamlandı; bu tarihsel config'in durum
  metni koşu kimliğini korumak için değiştirilmedi. Güncel gerçekleşen durum:
  [clean training raporu](../docs/clean-training-status.md). Gerekçeler:
  [eğitim protokolü](../docs/training-protocol.md).
- `training-augmented.json`: aynı sayısal training tarifini, tamamlanmış clean
  koşunun kimliğini, üç epoch'a ait denetlenmiş noisy view konumunu ve iki sabit
  checkpoint slotunu tanımlar. Kayıtların yarısı noisy view alır; gerçekleşen
  kelime değişimi yaklaşık %7,12–7,18'dir. Uygulama ve çalıştırma durumu:
  [augmentation hazırlığı](../docs/augmentation-preparation.md).
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
anlamına gelmez. Augmented komutunun ayarları `training-augmented.json` içindedir; mevcut
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

## Augmented fine-tuned model evaluation

`augmented-noise-evaluation.json` pins the completed augmented run's selected
epoch-03 export and tensor digest. It uses the same prepared validation noise,
runtime, batching and scoring as the clean/published pilots. No noise is
regenerated and no training is performed:

```bash
python scripts/evaluate_noise.py --config configs/augmented-noise-evaluation.json
```

Comparison scope and execution status:
[augmented-noise-pilot.md](../docs/augmented-noise-pilot.md).

## GPU severity and corruption-seed grid

`robustness-grid.json` freezes the 117-evaluation matrix and verified validation
transfer SHA. The three `{published,clean,augmented}-robustness-grid.json` files
pin model identities and share the same 38 noisy conditions plus fresh clean
controls. They require CUDA, a matched calibration receipt and one shared GPU
runtime/numerical policy. Historical CPU clean scores are diagnostics only.
Mean/sample SD varies corruption seeds, not the single training seed.
Execution and output map: [robustness-grid.md](../docs/robustness-grid.md).

## Case-loss training ablation

`ablation-case-loss.json` pins the omitted family, shared initialization,
completed reference runs, separate calibration/formal output directories and
five post-training validation conditions. It references the byte-identical
`training-augmented.json`; it does not change the paired numerical recipe.
`ablation-case-loss-inputs.json` records the 31-file input ZIP SHA and epoch
content/audit hashes. The omitted-family views are prepared locally once and
restored on Colab. See [preparation](../docs/case-loss-ablation-preparation.md).

## Held-out evaluation protocol

`held-out-test.json` freezes the four selected model identities, validation-derived
122-evaluation matrix, original noise recipe, two reporting cohorts and matched
inference/metric rules before test scoring. The 14,600-row primary cohort remains
the released clean-v1 eligible subset; the 14,445-row exact-input-overlap
sensitivity view filters those same predictions without additional inference.

`python scripts/freeze_heldout_protocol.py` verifies model-selection metadata,
pinned sources and original row IDs, then snapshots the protocol/config/row masks.
It reads test inputs for structural auditing but computes no predictions/scores.
This config is a protocol plan, not an input to the validation-only evaluator.
Separate test preparation and orchestration are implemented; actual GPU
readiness completed on 2026-10-23; all 122 formal conditions and the Colab
saved-prediction audit of 244 cohort score sets subsequently completed.
`held-out-execution.json` binds the unchanged protocol
hash/freeze identity to the new input/output paths, fixed model order and reused
validation probe manifest. It keeps test scoring off by default. This execution
wiring does not edit the frozen protocol's historical phase fields.
See [protocol](../docs/held-out-test-protocol.md) and
[freeze evidence](../docs/held-out-test-preparation.md), plus
[input/execution preparation](../docs/held-out-test-execution.md) and
[actual GPU launch](../docs/held-out-test-launch.md).

The configs and frozen numerical sources remain unchanged after scoring.
Completed analysis, local verification and result archive locations are recorded
in [held-out completion](../docs/held-out-test-completion.md);
[all condition results](../docs/held-out-test-results.md) keep validation/test distinct.
