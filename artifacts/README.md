# Local run outputs

Checkpoint, tahmin, ara çıktı ve deney günlükleri burada tutulacak ve Git'e
eklenmeyecek. Raporda kullanılan seçilmiş küçük tablolar ve grafikler daha sonra
`docs/` altında sürümlenebilir.

- `model-preview.json`: elle yazılmış üç örneğin tahminleri ve kelime skorları.
- `evaluations/published-bert-clean-validation-v1/metrics.json`: clean validation
  metrikleri, confusion matrix, ayarlar, kaynak/kod hash'leri ve çalışma bilgileri.
- Aynı klasörde `predictions.jsonl`: kaynak `row_id`, gerçek etiket ID'leri ve
  tahmin ID'leri; `error_examples.json`: kaynak sırasından seçilen az sayıda hata.
  Etiket ID eşlemesi `metrics.json` içinde kayıtlıdır. Bu yerel çıktıları Git'e ekleme.
- `source_snapshot/`: ölçümde kullanılan kod/ayarların sekiz kopyası; hash'ler bu
  arşivdeki sürümleri tanımlar. Çalışma ağacı sonradan değişse de ölçüm kodu korunur.

## Noise pilotu

`evaluations/published-bert-noise-pilot-v2/`:

- `run.json`: ayarlar, model ve temiz referans, kod/config/girdi hash'leri, paket sürümleri.
- `source_snapshot/`: bu ölçümde kullanılan kod/ayar kopyaları.
- `progress.json`: çalışma sürerken güncel koşul/kayıt sayısı; bittiğinde complete durumu.
- `conditions/<kosul>/predictions.jsonl`: aynı kelime konumlarının gerçek ve tahmin ID'leri.
- `conditions/<kosul>/metrics.json`: skorlar, temize göre düşüşler, eşleştirilmiş hata
  sayımları ve kullanılan noise koşulunun veri kontrolü raporu.
- `conditions/<kosul>/error_examples.json`: temiz ve gürültülü metin/tahminlerin yerel örnekleri.
- `summary.json`: yalnızca bütün koşullar tamamlandığında oluşan ortak sonuç.
- `analysis/analysis.json`: yeni hataların doğrudan değişen kelimede veya değişen
  bağlamda oluşma sayıları; girdi/tahmin ve analiz kodu hash'leri.
- `analysis/comparison.md`: tamamlanmış sonuçlardan üretilen karşılaştırma tabloları.
- `analysis/analyze_noise.py`: bu analizi üreten kaynak dosyanın kopyası.

`conditions/clean-anchor/`, bütün noise koşullarıyla aynı temiz uzunluk sıralamasını
kullanan yeni temiz kontrolü içerir. Eski temiz skorlarla fark da kaydedilir.
Tahminler bu değerlendirme sırasındadır; kaynak kayıt ID'leri korunur. Hata örnekleri
orijinal kaynak sırasındaki en erken örneklerden seçilir.

Temizken doğru olup yanlışlaşan ve yanlışken doğruya dönen kelimeler ayrı sayılır.
Girdi değişmeyen kayıtlarda tahmin değişimi ayrıca izlenir. Bir koşul ancak tamamen
hesaplandıktan sonra tamamlanmış klasörüne taşınır. Kesinti sonrasında `--resume`,
aynı kod/ayar/girdi hash'lerini ve tamamlanmış çıktı hash'lerini doğrular; mevcut
tamamlanmış koşulları yeniden hesaplamaz. Aynı koşuya eşzamanlı ikinci süreç engellenir.

`published-bert-noise-pilot-v1/`, kaynak sırasıyla başlayan ve hiçbir noise koşulu
tamamlanmadan durdurulan ilk girişimin kod/ayarlarını ve kesinti nedenini saklar.
Padding hesabına göre alınan hızlandırma kararı noisy aggregate skorlar hesaplanmadan
verildi; tamamlanmış sonuçlar v2 klasöründedir.

## Training ve GPU calibration

- `initializations/training-v1/seed-586/`: pinned BERTurk encoder ve yeni head'in
  ortak başlangıç ağırlıkları, tokenizer ve hash manifesti için yerel varsayılan yol.
  İlk calibration bu dosyaları Colab'ın geçici diskinde oluşturdu; ağırlıklar yerel
  rapor arşivine dahil edilmedi. Formal training öncesi kalıcı depolamaya alınmalı.
- `training/training-v1/calibrate-clean-seed-586/`: kısa GPU denemesinin ayarları,
  kaynak kopyaları, loss/LR/timing event'leri ve `calibration.json`. Güncellenmiş
  calibration ağırlıkları saklanmaz; süre tahmini validation/checkpoint I/O'yu kapsamaz.
- `training/training-v1/calibration-clean-seed-586-v1/`: 2026-10-18'de tamamlanan
  T4 denemesinden indirilen gerçek raporlar: `calibration.json`, `run.json`,
  `events.jsonl`, `progress.json` ve `source_snapshot/`. Tam olarak 30 update;
  kaynak snapshot hash'leri ve run identity yerelde doğrulandı.
- `training/training-v1/train-clean-seed-586/`: `run.json`, `events.jsonl`,
  `progress.json`, `source_snapshot/`, `checkpoints/`, `models/epoch-XX/` ve
  `validation/epoch-XX/`. `summary.json` yalnızca tam training bittiğinde yazılır.
- Checkpoint, model/optimizer/scheduler ve RNG state yanında row permutation,
  next-row cursor ve epoch/selection geçmişini saklar. İki son sağlam state ve
  üç epoch model export'u korunur; yaklaşık 4 GB/koşu ve 0,44 GB/ortak başlangıç.

Colab notebook'u calibration için varsayılan olarak geçici disk kullanır ve rapor
indirme bağlantısı oluşturur. `USE_DRIVE=True` formal training çıktılarını Drive'a
yönlendirir; bu oturumda Drive mount yetki aktarımı hatası verdi. Kalıcı depolama,
full-epoch validation ve checkpoint/resume henüz çalıştırılarak doğrulanmadı.
Model export'ları Hugging Face formatında yüklenebilir. Tam training sonucu yok;
gözlenen kısa denemenin özeti [calibration raporunda](../docs/training-calibration.md).
