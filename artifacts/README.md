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

## Tamamlanan augmented koşu

`training/training-v1/train-augmented-seed-586-v1/completion-2026-10-22/`, 3 epoch /
10.968 update tamamlanan koşudan alınan metadata, source/hash/event denetimi,
arşiv receipt'i ve ekran görüntülerini tutar. `metadata/` altında run/summary/
progress, üç epoch'un metrics/export ve augmentation audit dosyaları bulunur.
Geçici audit hücresi notebook'tan kaldırıldı; çalıştırılan yardımcı kaynaklar
burada saklanır. `local-archive-verification.json` arşiv/dosya kontrolünü,
`local-model-verification.json` dört offline model yüklemesini ve üç epoch'un
kaydedilmiş tahminlerinden yeniden hesaplanan metriklerini doğrular.

Kullanıcının indirdiği `augmented-seed586-models-3cdf62cc3be4.zip`, 1.772.193.862 byte ve
SHA-256 `b93de2fd2c0335c487332fcdb2d8ab10e9aaaf3eba11c9e43d93a7b5cfb44378`
ile doğrulandı. Üç model ve ortak initialization dahil 58 kaynak üyeyi içerir;
optimizer state içermez. ZIP bu koşunun yerel artifact köküne kopyalandı;
doğrulanan açılmış kopyası `completed-archive/` altında durur. Downloads'taki
orijinal de korundu. Yeni eğitim veya model inference yapılmadı.

Kapsamı belirlenen dört Drive öğesi, yerel doğrulama ve kullanıcının işlem anındaki
onayı sonrası kalıcı silindi: iki slotu içeren checkpoints klasörü, epoch-01/02
ve geçici ZIP. Epoch-03, raporlar ve ortak girdiler/initialization korundu.
`drive-cleanup-plan.json`, `drive-cleanup-result.json` ve ekran görüntüleri bu
işlemi kaydeder. Genel çöp kutusu boşaltılmadı. Son Drive göstergesi 15 GB'ın
2,75 GB'ının dolu olduğunu gösterdi (yaklaşık 12,25 GB boş). Güncel durum:
[augmented training raporu](../docs/augmented-training-status.md).

## Augmented medium-noise validation pilotu

`evaluations/finetuned-augmented-seed586-noise-pilot-v1/`, seçilen epoch-03'ün
clean control ve aynı dört medium/seed-586 validation koşulundaki tamamlanmış
ölçümüdür. Her koşul 14.453 kayıt / 179.164 kelime hedefini korur. `run.json`,
`summary.json`, `progress.json`, sabit source/config snapshot'ları ve her koşulun
prediction/metrics/error_examples dosyaları burada tutulur.

`analysis/` clean fine-tuned modelle, `analysis-published/` hazır modelle eşleşen
karşılaştırmayı içerir. Her ikisinde `analysis.json`, `comparison.md`,
`model-comparison.json`, `model-comparison.md` ve analiz kodunun snapshot'ı vardır.
Analiz yalnızca kaydedilmiş tahminleri okur; ek inference yapmaz.

`evaluation-preparation/augmented-seed586-noise-pilot-v1/` altında 45 eşleşme
kontrolünün `preflight.json` receipt'i, `evaluation.log`, `verify_results.py` ve
geçen `result-verification.json` denetimi bulunur. Beş koşulun dosya hash'leri,
ham validation hedefleri, sıra ve bütün metrikleri yeniden doğrulandı. Yeni noise,
training veya test değerlendirmesi yapılmadı. Toplu sonuçlar:
[augmented noise pilotu](../docs/augmented-noise-pilot.md).

## GPU robustness grid hazırlığı

`evaluation-preparation/robustness-grid-gpu-v1/`, yeniden üretilmeyen validation
girdilerinin 285 dosyalık ZIP'ini, hash receipt'ini, Drive yükleme kaydını ve ekran
görüntülerini tutar. ZIP yalnızca mevcut clean/noisy validation girdilerini,
ham validation kaynağını ve hazır modelin tarihsel clean referansını taşır;
training/test metni veya model ağırlığı içermez. Model export'ları proje Drive'ında
korunan seçili epoch-03 klasörlerinden okunur.

Tam GPU sonuçları hem proje Drive'ındaki hem yereldeki `artifacts/evaluations/`
altında üç ayrı model koşusunda tamamlandı. Ortak `robustness-grid-gpu-v1/`
kalibrasyon, ilerleme, execution log ve analiz çıktılarını tutar. `complete.json`
koşul/analiz dosyalarının doğrulanmış yayınını tanımlar. 533 dosyalık sonuç ZIP'i
yerelde hash/boyut ve güvenli çıkarma kontrollerinden geçti; 117 ölçümün metrikleri
ham hedeflerden yeniden hesaplandı, üç cross-model karşılaştırması ve 13 analiz
dosyası doğrulandı. ZIP yaklaşık 15,7 MB; model ağırlığı içermez.
`evaluation-preparation/robustness-grid-gpu-v1/` içindeki
`completed-result-verification.json` yerel doğrulama kaydıdır. Drive'daki sonuçlar
ve küçük arşiv korunuyor. Güncel çalışma durumu:
[GPU grid raporu](../docs/robustness-grid.md).

## Case-loss ablation hazırlığı

`training/training-v1/ablation-without-case-loss-preparation-seed-586/`, 31 dosyalık
girdi ZIP'ini, transfer receipt'ini ve paired view audit'ini tutar. Asıl üç epoch
view `data/processed/training-augmentation-v1/seed-586-without-case_loss/` altında;
mevcut tam augmentation ile row mask, korunan word plan ve operation eşleşmeleri
doğrulandı. Girdiler yeniden üretilmeden Drive'a aktarıldı ve tekrar doğrulandı.

`calibration/` içindeki dört calibration kaydı ve `readiness.json`, Colab'dan
alınan beş küçük dosyadır. `colab-ready-transfer.json` tam metin/hash zarfını,
`colab-preparation-verification.json` yerel identity/source/config/runtime ve
finite-loss denetimini saklar. T4 üzerinde 30 update tamamlandı; en uzun prepared
batch dahil edildi, calibration ağırlıkları saklanmadı. Bunlar model performansı
veya tamamlanmış bir ablation training sonucu değildir.

`live-*.py` yürütülen hazırlık ve kapanış hücrelerini; `preparation-ready.jpg`
son notebook durumunu, `files-changed.md` bu görevdeki tam dosya envanterini tutar.
Notebook'taki çalışma anahtarları kapalı; formal ablation başlamadı ve çalışan
iş yok. [Hazırlık raporu](../docs/case-loss-ablation-preparation.md).
