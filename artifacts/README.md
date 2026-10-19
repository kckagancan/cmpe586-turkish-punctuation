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
  rapor arşivine dahil edilmedi. Hazırlık komutu bu ortak initialization'ı Drive'daki
  proje klasörüne kopyalar ve tensor/file hash'lerini doğrular.
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

`preparation/training-v1/seed-586/`, `readiness.json` ve `pip-freeze.txt` hazırlık
kayıtları için yerel rapor kopyasıdır. Gerçek Colab hazırlığı ve input/init cache'i
`MyDrive/cmpe586-turkish-punctuation` altında tutulur. Hazırlık training update'i
veya resumable training checkpoint'i üretmez.

Colab notebook'unda bir mount-if-needed hücresi kalır; eski izin denemeleri ve
calibration tekrar hücreleri kaldırılmıştır. Drive mount başarıyla gözlendi.
Full-epoch validation ve checkpoint yazımı çalıştı; checkpoint'ten resume henüz
çalıştırılarak doğrulanmadı.
Model export'ları Hugging Face formatında yüklenebilir. Clean training 3 epoch /
10.968 update ile tamamlandı; metadata kopyaları
`training/training-v1/train-clean-seed-586-v1/` altında tutulur. Üç epoch ağırlıkları
yerelde arşivlidir; seçili epoch-03 ayrıca Drive'da kalır. Epoch skorları ve süre
[training raporunda](../docs/clean-training-status.md),
önceki kısa denemenin özeti [calibration raporunda](../docs/training-calibration.md).

Kullanıcının normal tarayıcıyla indirdiği `clean-seed586-5ddecc9b8bfa.zip` ayrıca
`training/training-v1/train-clean-seed-586-v1/` altında saklanır; doğrulanmış model
`recovery/models/epoch-03/` içindedir. Bu yedek büyük optimizer checkpoint'lerini
ve epoch-01/02 ağırlıklarını içermez. Denetim scriptleri, JSON ve test logları aynı
artifact klasöründedir: [yedek denetimi](../docs/backup-verification.md).

Tamamlanan clean koşunun ek model/sonuç arşivi
`training/training-v1/train-clean-seed-586-v1/clean-seed586-models-5ddecc9b8bfa.zip`
adıyla yerelde doğrulandı. Açılmış kopyası `completed-archive/` altındadır:
`initialization/` ortak başlangıcı, `run/models/epoch-{01,02,03}/` üç epoch
modelini; `run/validation/` ve metadata/source dosyaları deney kayıtlarını tutar.
49 kaynak dosyanın hash'leri, run manifest'leri ve dört modelin offline yüklenmesi
doğrulandı. Optimizer `state.pt` dosyaları ve ham veri bu ZIP'e dahil değildir.
Ham veri zaten ayrı yerel data klasöründedir. Her başarılı training sonrası tek
ZIP hazırlanır, kullanıcı manuel indirir, yerel doğrulamadan sonra belirlenen
Drive kopyaları temizlenir. Ağırlıklar ve arşiv Git'e eklenmez.

Bu koşunun Drive temizliği tamamlandı: iki optimizer checkpoint'i, epoch-01/02
kopyaları ve geçici transfer ZIP'i onay sonrası kalıcı silindi. Epoch-03, raporlar
ve ortak girdiler/initialization korundu. Drive yaklaşık 10 GB boş gösteriyor.
`drive-cleanup-plan.json`, `drive-cleanup-result.json` ve `drive-cleanup-*.jpg`
dosyaları kimlikleri ve sonucu kaydeder. Bunlar da yerel, Git dışında kalan
artifact dosyalarıdır; hesabın genel çöp kutusu boşaltılmadı.

## Augmented hazırlık kayıtları

`training/training-v1/augmented-preparation-seed-586/`, üç noisy epoch view'ın
yerel manifestini ve Colab hazırlık/calibration/storage kontrolünden alınan
kanıtları tutar. `local-checkpoint-inspection/` gerçek model yüklemeden dört küçük
yazma/yükleme yapar; sonunda iki `state.pt` slotu kalır. Bunlar optimizer resume
veya Drive'ın bulut tarafındaki kalıcılığı için kanıt değildir.

Asıl epoch view'ları `data/processed/training-augmentation-v1/seed-586/` altında
kalır; toplam yaklaşık 221 MB'tır. Colab kopyası yalnızca projenin Drive input
klasörüne yazılır. Calibration ayrı çıktı klasörü kullanır ve güncellenen model
ağırlıklarını saklamaz. Gerçekleşen adımlar ve sınırlar:
[augmentation hazırlık raporu](../docs/augmentation-preparation.md).

Bu klasördeki `calibration/` Colab'ın gerçek `run.json`, `calibration.json`,
`events.jsonl` ve `progress.json` dosyalarını tutar. `colab-view-manifest.json`,
`colab-readiness.json`, `drive-checkpoint-inspection.json` ve transfer receipt'i
de yerelde hash kontrolünden geçti. GPU calibration 30 update ile tamamlandı;
full augmented training başlamadı. Bu küçük dosyalar ve ekran görüntüleri de
Git dışında kalır.
