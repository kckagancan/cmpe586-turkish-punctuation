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
Hazırlık aşamasındaki duruşta çalışma anahtarları kapalıydı ve formal ablation
başlamamıştı. [Hazırlık raporu](../docs/case-loss-ablation-preparation.md).

## Case-loss ablation tamamlanma ve arşiv

`training/training-v1/train-augmented-without-case-loss-seed-586-v1/launch/`
ilk 25 update/run-identity kontrolünü ve başlangıç runtime/readiness kayıtlarını
saklar. Formal koşu 2026-10-22'de tamamlandı: üç epoch / 10.968 update,
seçilen epoch 3, clean validation macro-F1 %84,1310.

Aynı koşunun `completion-2026-10-22/` klasöründe küçük metadata kopyaları,
`completion-verification.json`, `archive-receipt.json`, denetim kaynakları ve tam
`files-changed.md` envanteri tutulur. İlk denetimin kaynak-sırası varsayımı
düzeltilmiştir; başarısız ilk kaynak ve düzeltme açıklaması da korunur. Son
denetim, runner'ın sabit uzunluk sırasını ve ham validation hedeflerini doğrular.
Tüm epoch skorları yeniden hesaplanmıştır; yeni inference yapılmamıştır.

Model/sonuç ZIP'i `ablation-without-case-loss-seed586-models-9a74ed0ee701.zip`
adıyla Drive'da hazırdır: 1.772.191.681 byte, 58 kaynak dosya ve manifest.
ZIP/member ve Drive readback hash kontrolleri geçti. Kullanıcının indirdiği
ZIP'in hash'i eşleşti; koşu artifact köküne kopyalanıp `completed-archive/`
altına güvenli çıkarıldı. **Dört model offline yüklendi**; finite tensor hash'leri,
tokenizer ve etiket eşlemeleri doğrulandı. Üç epoch'un kaydedilmiş skorları da
yerel ham validation hedeflerinden yeniden hesaplanıp eşleştirildi.

`local-archive-verification.json` ve `local-model-verification.json` bu denetimleri
saklar. Ardından yalnızca bu koşunun 19 dosyası temizlendi: iki optimizer slotu,
epoch-01/02 Drive model dosyaları ve geçici ZIP. Boş klasörler kalır; epoch-03,
metadata ve ortak kaynaklar Drive'da korunur. Kapsam, tekil işlem yanıtları ve
sonraki listelemeler `drive-cleanup-candidate.json`, `drive-cleanup-execution.json`
ve `drive-cleanup-result.json` içinde. Optimizer state'ler tamamlanmış koşu
arşivine dahil değil. Arşivleme aşamasında noisy ablation değerlendirmesi yoktu;
sonraki değerlendirme aşağıda izlenir. Held-out test yapılmadı.
[Tamamlanma raporu](../docs/case-loss-ablation-training-status.md).

## Ablation evaluation preparation receipts

`evaluation-preparation/ablation-case-loss-medium-gpu-v1/` stores the ignored
local source-transfer bundle, exact Colab readiness/deployment/launch readbacks,
initial observation log, local source verification and a complete local inventory.
The completed result run is restored locally at
`evaluations/finetuned-augmented-without-case-loss-seed586-medium-gpu-v1/`.
Its `analysis/` and condition commits remain byte-identical to Drive.
`completion-2026-10-22/` contains 62 exact Drive result/preparation files,
the transfer receipt, `local-verification.json`, `report-generation.json`,
complete inventory and the 730,799-byte results-only ZIP. The ZIP SHA-256 is
`5a2e803a4a08ec13816320765d53ace4b24d0915b51c44ab3a29531af5e4bfd0`.
All 20 matching saved condition metrics passed local raw-target recomputation;
no inference, training, weights download or Drive deletion occurred in finalization.
The previously verified 1.77 GB training/model archive remains separate.
See [method and execution](../docs/case-loss-ablation-evaluation.md).

## Held-out protocol freeze

`evaluation-preparation/held-out-test-v1/protocol-freeze-r2/` contains the immutable
protocol/config/source/model-metadata snapshots, `freeze.json`, `cohorts.json`,
aggregate counts and last-written `complete.json`. Freeze identity:
`ba6bf8decbd62a64f898a4eb47054776a97b3131fc8c6a20d8a50e22c6dc9653`.
Primary test IDs: 14,600; sensitivity IDs: 14,445, excluding the union of 155
exact train/validation input overlaps. Full row IDs remain ignored; only the
aggregate summary is published under `docs/`.

Structural source and model-selection metadata were audited without loading or
rehashing weights. No test noise, predictions, scores or training outputs were
created. Operational receipts and complete inventory belong in the sibling
`protocol-preparation-2026-10-22/` directory, leaving the committed freeze intact.
See [protocol preparation](../docs/held-out-test-preparation.md).

## Held-out input and execution preparation

`data/processed/noise-held-out-test-v1/` (at the repository root) contains 39
audited test views, 274 files plus the immutable hash commit. Every view retains
14,600 rows / 181,485 original word targets; independent re-tokenization passed.
No model predictions or scores existed at that local preparation handoff.

`evaluation-preparation/held-out-test-v1/transfer-v1/` contains the 72,979,242-byte
inputs ZIP (312 members + manifest), 74,440-byte source ZIP (33 + manifest), and
`transfer.json`. Full ZIP/member hashes and restoration against existing local
bytes passed; models are not included. Preparation/verification/transfer logs
are in the parent directory; `input-execution-2026-10-22/` stores final checks and
the complete inventory. The protocol freeze remains unchanged.

Future Drive readiness: `preparation/held-out-test-v1/runtime-v1/`.
Future inference: `artifacts/evaluations/held-out-test-v1/`; future saved-prediction
analysis: its `analysis-v1/`. These future paths are not evidence of completed
runs. No actual GPU readiness, test inference, upload or Drive cleanup ran in
this preparation. [Execution details](../docs/held-out-test-execution.md).

## Held-out GPU launch — 2026-10-23

`evaluation-preparation/held-out-test-v1/launch-2026-10-23/` contains the observed
Drive deployment IDs, original failed restoration source/log, corrected
restoration receipt/log, exact downloaded readiness/launch records and local
receipt/source/input verification. GPU readiness identity:
`f52980b75c4fafaac6138613f131eb1ca5d6b66a5436689fa4f424588793f2c7`.
All 39 full re-tokenizations, four model tensor identities and eight validation-only
GPU probes passed. No scientific recipe or historical snapshot changed.

Formal held-out PID 7592 started at 12:25:56 Istanbul time. The active remote
output is `artifacts/evaluations/held-out-test-v1/` under the project Drive root;
streaming logs and startup evidence are in its separate dated preparation folder.
This dated launch handoff is superseded by the completed run/archive below.
Initial observations and its original inventory remain in the dated launch folder.
See [GPU restoration and launch](../docs/held-out-test-launch.md).

## Held-out completed results — 2026-10-23

`evaluations/held-out-test-v1/` contains 526 exact downloaded Drive files: all
122 condition prediction/metric/example/commit sets, source/config provenance,
final progress and the original committed `analysis-v1/`. Run identity:
`60e31867535de346accb64be45df04984434f6ff328b31c8819115cb689538a0`.

`evaluation-preparation/held-out-test-v1/completion-2026-10-23/` contains the
15,826,627-byte original transfer ZIP, exact transfer receipt and Colab audit log,
local CPU `local-analysis/`, archive/report verification receipts and inventory.
All 122 conditions / 244 cohort scores were independently recomputed locally;
the analyses match except creation time and their aggregate CSV/Markdown match
byte for byte. No model load or new inference was required.

The final `held-out-test-v1-results.zip` has **570 files plus manifest**,
**16,011,054 bytes**, SHA-256
`2463a23ed22dc9543341393bedfb902f693448330da860374c708748d4830925`.
It includes predictions/results, frozen protocol/masks, readiness/launch receipts,
local audit log and verification source. Every member has a byte count and SHA
and passed readback. Model weights/optimizer state are excluded; existing verified
training archives remain separate. Original Drive evaluation copies were later
removed after verified local backup; see the scoped cleanup record below.

Only aggregate tables, condition summaries, diagnostic counts and figures go
under `docs/`; full texts, row masks, predictions and raw examples stay ignored.
[Results](../docs/held-out-test-results.md), [completion record](../docs/held-out-test-completion.md).

## Drive storage audit — 2026-10-23

`storage-audit/drive-2026-10-23-v1/` stores the complete metadata-only Drive tree
(424 folders / 1,439 files / 2,859,545,316 logical bytes), every file's classification,
and a proposed 1,148-file / 925,753,181-byte cleanup scope. Local candidate files
match verified archives by size/SHA; current Drive sizes match. Current remote
contents still require hash readback before any agreed cleanup. No Drive file
was changed or deleted; this inventory is not a quota measurement or deletion approval.

`augmented-training-inputs-seed586.zip` is a new local-only backup of the existing
31-file augmented training-view tree: 44,255,470 bytes, SHA-256
`0296e955bb772835d47bfe8529e317cf49fdb4f6cfde252f8e2b95c74d02652d`.
All original epoch/source hashes and ZIP member readbacks passed; no inputs were
regenerated and no models were loaded. Other verified input/result archives were
reused. `inventory.json` and `files-changed.md` list all new evidence and edits.
Models/shared initialization and every local archive remain protected.
See [inventory and concrete proposal](../docs/drive-storage-audit.md).

## Verified Drive cleanup — 2026-10-23

The initial audit above is historical. `storage-audit/drive-2026-10-23-v1/cleanup-2026-10-23/`
contains the authorized cleanup's complete evidence: exact remote content
hashes, both preflight metadata snapshots, manifest-resolution evidence,
versioned candidate list, local manifest supplement, deletion intents and
1,148 per-file receipts, post-cleanup metadata and verification, final checks
and complete `inventory.json` / `files-changed.md`.

The remote augmented preparation manifest differs from the local preparation
manifest only in runtime/timing fields. Its exact historically verified local
counterpart is preserved in the **9,705-byte supplement ZIP**, SHA
`edd9e56f606285f2d2a5e5c420f267f0dd00bacce920c2a8425ec90ed0cb036a`.
The original candidate manifest and local augmented-input ZIP remain untouched;
use the supplement to restore the exact original Drive manifest.

All 1,148 reviewed Drive file IDs / 925,753,181 bytes were removed. The fresh
post-inventory contains 292 files / 1,934,065,963 logical bytes; all 291 original
protected IDs plus the new remote readback receipt remain. All eight backup ZIP
hashes still match. Models/local results remain, and no experiment was repeated.
See [completed scoped cleanup](../docs/drive-cleanup.md).
