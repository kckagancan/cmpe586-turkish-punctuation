# CMPE 586 — Turkish Punctuation Restoration

**Robustness of Turkish Punctuation Restoration under Informal Writing Noise**

CMPE 586 Transformer/NLP ders projesi. Türkçe noktalama geri yükleme modellerinin
büyük/küçük harf kaybı, Türkçe karakterlerin sadeleştirilmesi ve yazım hataları
altındaki dayanıklılığı incelenecek.

**Durum:** Yerel ortam ve public repo hazır. Veri incelendi; `clean-v1` model
girdileri hazırlandı ve yayımlanmış BERT üç elle yazılmış örnekte çalıştırıldı.
Veri lisansı ve kesin kaynak/split geçmişi henüz doğrulanmadı.
Hazır modelin temiz validation ölçümü tamamlandı: noktalama macro-F1 **%83,64**.
Hazır modelin dört medium koşulluk validation pilotu tamamlandı; held-out test değerlendirmesi henüz yok.
Temiz ölçümün yöntemi ve sınıf sonuçları: [clean validation raporu](docs/clean-validation.md).
Gürültü kuralları, dönüşümler ve model çalıştırmadan veri hazırlama komutu hazır.
38 validation varyantının veri/hizalama kontrolü geçti: [noise veri raporu](docs/noise-audit.md).
Pilot macro-F1: case loss **%81,23**, Türkçe karakter kaybı **%83,43**, harf takası
**%82,90**, combined **%80,34**. Yöntem, sınıf skorları ve hata analizi:
[pilot raporu](docs/noise-pilot.md).
Kontrollü fine-tuning için başlangıç ayarları ve karşılaştırma kuralları belirlendi:
[eğitim protokolü](docs/training-protocol.md). Clean training pipeline ve Colab
notebook'u hazır. T4 üzerinde 30 update'lik GPU calibration tamamlandı;
[ölçümler ve sınırlar](docs/training-calibration.md). İlk clean fine-tuning
2026-10-19'da tamamlandı: 3 epoch / 10.968 update, seçilen epoch 3,
clean validation macro-F1 **%84,04**, training loop **72,2 dakika**.
Sonuçlar ve kapsam: [koşu raporu](docs/clean-training-status.md).
Clean modelin medium/seed-586 noise pilotu da tamamlandı: case loss **%81,23**,
diacritic loss **%83,80**, typo swap **%83,20**, combined **%80,10** macro-F1.
Clean control, training sonundaki 179.164 kelime tahminini aynen yeniden üretti.
Hazır modelle eşleşen kıyas, sınıf skorları ve hata analizi:
[clean model noise raporu](docs/clean-noise-pilot.md).
Koşunun bütün epoch modelleri ve sonuçları yerelde arşivlenip doğrulandı;
gereksiz Drive kopyaları temizlendi: [yedek denetimi](docs/backup-verification.md).
Augmentation uygulandı; üç epoch'un noisy view'ları, kelime hedefleri ve yeniden
tokenization kontrolleri yerelde ve Colab'da geçti. İki sabit checkpoint slotunun
küçük Drive kontrolü ve T4 üzerinde 30 update calibration tamamlandı
(peak allocated 2,434 GiB). Ayrıntılar ve kapsam:
[augmentation hazırlık raporu](docs/augmentation-preparation.md).
İlk formal augmented training 2026-10-22'de tamamlandı: 3 epoch / 10.968 update,
seçilen epoch 3, clean validation macro-F1 **%83,99**, loop süresi **82,9 dakika**.
Eşleşen medium-noise validation pilotu da tamamlandı: augmented macro-F1
case loss **%82,24**, diacritic loss **%83,97**, typo swap **%83,53**, combined
**%81,84**. Clean fine-tuning'e göre dört noisy koşulda +0,17–1,75 puan kazanç,
clean skorda −0,0485 puan fark gözlendi. Tek training/corruption seed'li bu pilot
held-out test sonucu değildir: [karşılaştırma ve hata analizi](docs/augmented-noise-pilot.md).
Model/sonuç ZIP'i yerelde arşivlendi; 58 dosyanın hash'i, dört modelin offline yüklenmesi ve kaydedilmiş
validation skorları doğrulandı. Gereksiz Drive kopyaları onay sonrası temizlendi:
[koşu ve arşiv durumu](docs/augmented-training-status.md).
Notebook'un training/calibration anahtarları kapalı; arşivleme de ayrı ve kapalı
bir hücrede. Özet hücresi, bağlantı yenilendiğinde kalıcı Drive sonucunu okuyabilir.

## Çalışma düzeni

- **VS Code + `.py`:** tekrar kullanılabilir veri, noise, eğitim ve değerlendirme kodu.
- **Notebook:** veri inceleme, küçük örnekler, görselleştirme ve Colab'dan çalıştırma.
- **Yerel CPU:** geliştirme ve küçük kontroller.
- **Colab GPU:** fine-tuning ve kapsamlı deneyler; GPU erişimi oturuma bağlıdır.

Notebook'lar `src/` içindeki ortak kodu kullanacak. Böylece yerelde geliştirilen
kod Colab'da da aynı şekilde kullanılabilecek.

## Yerel kurulum

Python 3.11 ve `cmpe586` Conda ortamı kullanılır. Projeyi ilk kez kurarken:

```bash
git clone https://github.com/kckagancan/cmpe586-turkish-punctuation.git
cd cmpe586-turkish-punctuation
conda env create -f environment.yml
conda activate cmpe586
python -m pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
python -m ipykernel install --prefix "$CONDA_PREFIX" --name cmpe586 --display-name 'Python (cmpe586)'
python scripts/check_setup.py
```

Ortam bu bilgisayarda zaten kurulmuşsa günlük başlangıç:

```bash
conda activate cmpe586
code .
```

VS Code'da **Python: Select Interpreter** üzerinden `cmpe586` ortamını seç.
Notebook açınca sağ üstteki kernel seçiminde aynı ortamı kullan.
Terminalde `conda activate` çalıştırmak editörün Python seçimini değiştirmeyebilir;
import uyarısı varsa seçili interpreter'ı ayrıca kontrol et.
Python ve Jupyter VS Code eklentileri önerilen eklentiler arasında tanımlı.

Ortamı yeniden kontrol etmek için `python scripts/check_setup.py` çalıştırılabilir.
`notebooks/01_data_inspection.ipynb` veri setinin tüm bölümlerini okur;
yapı kontrollerini, etiket dağılımlarını ve boş token'ları gösterir.
Ortak okuyucu ve doğrulama kodu `src/turkish_punctuation/data.py` içindedir;
notebook ve genel inceleme aracı aynı kodu kullanır.
Tarayıcı tabanlı Jupyter tercih edilirse `jupyter lab` çalıştırılabilir.

Tokenizer ve kelime–alt parça hizalaması incelemesi:

```bash
python scripts/inspect_tokenizer.py --download-tokenizer
```

Bu komut sabit sürümdeki BERT tokenizer dosyalarını `.cache/huggingface/` içine
indirir; model ağırlığı indirmez. Sonraki çalıştırmalarda çevrimdışı önbelleği
kullanmak için `--download-tokenizer` seçeneğini çıkar.
Sonuçlar `docs/tokenization-audit.json` içinde; yorumları
[veri inceleme raporunun tokenizer bölümünde](docs/data-audit.md#tokenizer-and-alignment-audit).
İnceleme hiçbir temizlenmiş eğitim/test bölümü oluşturmaz.

Model girdilerini hazırlamak için, tokenizer önbellekteyken:

```bash
python scripts/prepare_data.py
```

`configs/preprocessing.json` içindeki `clean-v1` politikası, boş kelime içeren
kaydın tamamını ayrı tutar. Ham dosyalar değişmez. Yerel çıktı
`data/processed/clean-v1/` altında oluşur: `dataset/` token ID'lerini ve hizalanmış
etiketleri; `quarantine/` ayrılan orijinal kayıtları; `manifest.json` kaynakları,
ayarları, kayıt ID'lerini ve sınıf sayılarını saklar. Mevcut çıktı üzerine yazılmaz.
Bu alt kümedeki skorlar, yayımlanmış tam test skorlarıyla doğrudan eşdeğer değildir.
Tüm modeller ve noise koşulları aynı uygun kayıtları kullanacak; noise sonrasında
yeni hizalama/uzunluk sorunu oluşursa kayıt sessizce atılmayacak, işlem duracak.

Ortak hazırlama ve DataLoader kodu `src/turkish_punctuation/preprocessing.py`
içindedir. Hugging Face `Dataset`/`DataCollatorForTokenClassification` ve PyTorch
`DataLoader` kullanılır. Padding batch içinde dinamik yapılır; padding etiketleri
`-100` olur. `row_id` hata analizi için saklanır ve model girdilerinden çıkarılır.
Hazırlama komutu her split'in ilk sıralı batch'ini oluşturup boyutlarını kaydeder;
bu adım model çalıştırmaz veya eğitim yapmaz. Önizlemedeki batch size 8, gelecekteki
GPU eğitimi için verilmiş bir batch size kararı değildir.

Yayımlanmış BERT modelini ilk kez indirip üç elle yazılmış örnekte çalıştırmak için:

```bash
python scripts/inspect_model.py --download-model
```

Sabit sürümdeki ağırlık dosyası yaklaşık 440 MB'tır ve `.cache/huggingface/` altında
saklanır. Sonraki çalıştırmalarda `--download-model` olmadan yerel önbellek kullanılır.
`src/turkish_punctuation/inference.py`, checkpoint'in sınıf eşlemesini ve ağırlıkların
eksiksiz yüklenmesini kontrol eder; tahmini `eval()` ve `torch.inference_mode()` ile
yapar. Kelime başına ilk subword'ün dört sınıf skoru (logits) arasından en büyüğü
seçilir. Model girdilerine referans etiketler gönderilmez.

Örnek tahminler, kelime düzeyindeki logits ve mimari bilgileri yerel
`artifacts/model-preview.json` dosyasına yazılır. Bu küçük örnekler veri setinden
seçilmemiştir ve model başarımını ölçmez. Gerçek validation ölçümü aşağıdaki
değerlendirme komutuyla yapılır. Tamamlanan noise pilotu aşağıda ayrıca açıklanır;
bu bölüm yayımlanmış modelin ilk ölçümlerini anlatır. Kendi fine-tuning sonucumuz
[clean training raporunda](docs/clean-training-status.md).

Hazırlanmış clean validation bölümünü yayımlanmış modelle değerlendirmek için:

```bash
python scripts/evaluate.py
```

Bu komut yalnızca validation bölümünü değerlendirir ve yerel tokenizer/model
önbelleğini kullanır. Ayarlar `configs/evaluation.json` içindedir. Ana skor,
period/comma/question mark sınıflarının eşit ağırlıklı macro-F1'ıdır; micro-P/R/F1,
sınıf bazında P/R/F1 ve dört sınıflı confusion matrix de kaydedilir. `O` konumları
sayımda korunur: noktalama olmayan bir konuma işaret eklemek yanlış pozitiftir.
Sıfır paydalarda skor 0'dır; padding ve devam subword'leri sayılmaz.

Çıktılar `artifacts/evaluations/published-bert-clean-validation-v1/` altına yazılır:
`metrics.json`, `predictions.jsonl`, `error_examples.json`. Mevcut çıktı üzerine
yazılmaz; tekrar ölçüm için ayarlarda yeni run adı ve çıktı yolu kullan.
Sonuçları anlamlandırırken bu ölçümün uygun validation alt kümesine ait olduğunu,
hazır checkpoint'in geçmişte aynı validation ile seçilmiş olabileceğini dikkate al.
Bu çalışma fine-tuning veya yayımlanmış tam test skorunun yeniden üretimi değildir.

## Gürültü protokolü

[Noise protokolü](docs/noise-protocol.md) ve [ayarları](configs/noise.json),
literatürden esinlenerek bu görevin kelime–etiket hizalamasına göre tasarlandı.
Tekil koşullar: Türkçe kurallarla küçük harfe çevirme, belirli Türkçe harfleri
sadeleştirme ve kelimenin içindeki iki komşu harfi bir kez takas etme.
Mild/medium/heavy seviyelerinde her tür için **tüm kelimelerin %2/%5/%10'u**
değiştirilir; oran yalnızca uygun kelimeler üzerinden hesaplanmaz.

Birleşik koşul aynı bileşen seçimlerini kullanır; daha fazla kelimeyi etkileyebilir.
Tam küçük harf ve tam Türkçe karakter sadeleştirme ayrıca değerlendirilir.
Seed'ler ve seçim kuralları ayarlarda kaydedilir. Kelime konumları ve etiketler
korunur; her varyant yeniden tokenize edilip hizalanır. Hazırlama komutu:

```bash
python scripts/prepare_noise.py
```

Bu komut yalnızca sabit uygun validation kayıtlarından 36 seed'li koşul ve iki
full koşul üretir. Tokenizer yerel önbellekten yüklenir; model ağırlığı yüklenmez.
Çıktı `data/processed/noise-v1/` altında kalır: her koşulun kelimeleri, dönüşüm
günlüğü, model girdileri, küçük örnekleri ve veri kontrolü raporu saklanır.
Manifest ayarları, kaynak/kod hash'lerini ve kullanılan kodun kopyalarını tanımlar.
Hizalama/uzunluk sorunu olursa işlem durur; yeni kayıt atılmaz veya kırpılmaz.

Tekrar denemek için JSON ayarını kopyala, yeni çıktı yolu belirle ve
`python scripts/prepare_noise.py --config configs/yeni-noise-ayari.json` kullan.
Desteklenmeyen kurallar ve uygulanamayacak oranlar açık hata verir.
Küçük validation pilotu tamamlandı; sonuçlar [pilot raporunda](docs/noise-pilot.md).
Eğitimdeki noise karışımı [training protokolünde](docs/training-protocol.md) tanımlıdır.
Tamamlanan veri kontrolünde kayıp kelime, `[UNK]` veya uzunluk aşımı bulunmadı;
en uzun girdi 218 token oldu. Ayrıntılar: [noise veri raporu](docs/noise-audit.md).

## Gürültülü model değerlendirmesi

```bash
python scripts/evaluate_noise.py
```

`configs/noise-evaluation.json`, dört medium/seed-586 koşulunu seçer. Komut temiz
tahmin dosyasının hash'ini doğrular ve skorlarını yeniden hesaplar; her noise
koşulunu yeniden tokenize edip kaydedilmiş girdilerle karşılaştırır. Model/tokenizer,
kayıtlar, hedefler, paket sürümleri ve çıkarım ayarları temiz referansla eşleşir.
Model yerel önbellekten bir kez yüklenir; eğitim veya test değerlendirmesi yapılmaz.
Padding maliyetini azaltmak için kayıtlar temiz token uzunluğuna göre sıralanır;
bütün koşullar aynı sıralamayı kullanır. Önce temiz kontrol aynı sırayla yeniden
ölçülür ve eski temiz tahminlerle karşılaştırılır. Noise farkları bu eşleşen temiz
kontrole göre hesaplanır; hiçbir kayıt atılmaz veya kırpılmaz.

Sonuçlar `artifacts/evaluations/published-bert-noise-pilot-v2/` altında tutulur:
her koşulun tahminleri, metrikleri, temize göre yüzde puan farkları ve eşleştirilmiş
hata örnekleri kaydedilir. `summary.json` bütün koşulları bir araya getirir.
`conditions/clean-anchor/` yeni temiz kontrolü saklar. Tahmin dosyaları değerlendirme
sırasındadır; her satırın kaynak `row_id` değeri korunur.
Örnekler, temizken doğru olup bozulan, yanlışken düzelen ve farklı bir yanlış
tahmine dönüşen konumları içerir; kaynak sırasından seçilir ve temsilî örneklem değildir.

Tamamlanmış çıktı üzerine yazılmaz. Kesilmiş aynı koşuya devam etmek için
`python scripts/evaluate_noise.py --resume` kullanılır; kod/ayar/veri değiştiyse
yeni çıktı yolu gerekir. `progress.json` devam eden koşulu ve tamamlanan kayıtları
gösterir. Bu ölçüm aynı yayımlanmış checkpoint'in validation dayanıklılığıdır;
augmentation eğitiminin katkısı daha sonra kontrollü eğitimlerle ölçülecek.

Tamamlanmış pilotun eşleştirilmiş hata analizini üretmek için
`python scripts/analyze_noise.py` kullanılır. Model yeniden çalışmaz; yerel
`analysis/` altında sayımlar, karşılaştırma tablosu ve analiz kodunun kopyası
saklanır. Mevcut analiz üzerine yazılmaz; yeni analiz için `--output-directory`
seçilir. Temiz kontrolün 179.164 kelime tahmini eski referansla aynı çıktı.

## Fine-tuning planı

[Eğitim protokolü](docs/training-protocol.md) ve [training.json](configs/training.json),
clean/augmented karşılaştırmasının başlangıç ayarlarını tanımlar: pinned BERTurk
temel ağırlıkları ve yeni dört sınıflı head, LR 2e-5, 3 epoch, effective batch 32
(8 × 4 accumulation), AdamW ve temiz validation macro-F1 ile checkpoint seçimi.
Augmented kolda aynı kayıtların yarısı her epoch yenilenen combined-medium
noisy input olarak sunulacak; veri boyutu ve kelime hedefi sayısı sabit kalacak.

Clean training kodu ve [Colab notebook'u](notebooks/02_clean_training_colab.ipynb)
hazır. T4 üzerinde 30 update'lik calibration başarıyla tamamlandı: batch 8 ×
accumulation 4, ortalama 0,476 s/update, peak allocated 2,43 GiB. Üç epoch için
yalnızca update işi yaklaşık 87 dakika tahmin ediliyor; validation ve checkpoint
I/O buna dahil değil. [Ayrıntılı rapor](docs/training-calibration.md).
Clean training tamamlandı. Augmented hazırlayıcı ve ayrı
[Colab notebook'u](notebooks/03_augmented_training_colab.ipynb) hazır;
[güncel hazırlık durumu](docs/augmentation-preparation.md) ayrıca kaydedilir.
Komut varsayılan olarak yalnızca 30 update'lik
calibration yapar ve CUDA ister:

```bash
python scripts/train.py --mode calibrate --download-model
```

Tam clean training ve kesilmiş aynı koşuya devam etmek için:

```bash
python scripts/train.py --mode train --download-model
python scripts/train.py --mode train --download-model --resume
```

Colab notebook'u artık yalnızca setup, tek Drive bağlantısı, kalıcı hazırlık,
hazırlık durumu ve ayrı kapalı training hücresini içerir. Drive zaten bağlıysa
yeniden `drive.mount()` çağırmaz. İlk calibration sırasındaki bağlantı sorunu
sonrasında başarılı mount gözlendi. Dosya işlemleri yalnızca
`MyDrive/cmpe586-turkish-punctuation` altında yapılır.

`scripts/prepare_training.py`, geçici oturumdaki initialization, input cache ve
calibration raporunu bu klasöre hash kontrolüyle kopyalar; farklı mevcut
klasörleri ezmez. Yeni oturumda kalıcı cache'i geri yükler ve train/validation ile
başlangıç ağırlıklarını yeniden doğrular. **Hazırlık sıfır training update'i yapar**;
bir optimizer veya training checkpoint'i oluşturmaz. `readiness.json` ve gerçek
paket sürümleri kaydedilir. Calibration `fc17d41` commit'inde çalışmıştır; önceki
ölçümün kaynak snapshot'ı korunur. Yeni notebook calibration'ı tekrar başlatmaz.
Calibration ve tam training ayrı çıktı klasörleri kullanır. Calibration modelinin
güncellenmiş ağırlıkları saklanmaz; formal training ortak initialization'dan başlar.
Resume, aynı kod/config/girdi/paket/GPU kimliğini ve checkpoint hash'lerini ister.
Kesintiden sonra son sağlam checkpoint'ten sonraki update'ler tekrar yürütülür.
Notebook'ta tam training ayrı bir hücrede varsayılan olarak kapalıdır.
Kalıcı hazırlık, training update'i çalıştırılmadan tamamlandı:
[hazırlık durumu](docs/preparation-status.md).
İlk formal clean training koşusu 2026-10-19'da tamamlandı; epoch skorları,
süre ve model yolları [training raporunda](docs/clean-training-status.md) kayıtlıdır.

## Colab

GPU runtime açıldıktan sonra yeni oturumda repo ve proje kurulumu:

```python
!git clone https://github.com/kckagancan/cmpe586-turkish-punctuation.git
%cd /content/cmpe586-turkish-punctuation
%pip install -e '.[dev]'
!python scripts/check_setup.py
```

Colab'ın hazır GPU PyTorch kurulumunu kullan; yerel CPU kurulum komutunu Colab'da
çalıştırma. Bağımlılık kurulumu yüklü paketleri değiştirirse runtime'ı yeniden
başlatmak gerekebilir. İlk calibration, runtime **2026.07**, Python **3.12.13**,
PyTorch **2.11.0+cu128** ve **T4** ile çalıştı. Proje Python 3.11/3.12 destekler;
runtime'ın gerçek sürümünü kontrol et.
Notebook kurulumu `requirements/colab.constraints.txt` ile gözlenen ana paket
sürümlerini kullanır; bu tam bir environment lock değildir ve Colab'ın GPU
PyTorch kurulumunu değiştirmez. Hazırlık raporu ayrıca tam `pip freeze` çıktısını
saklar. Notebook test split'ini indirmez/hazırlamaz ve training anahtarı varsayılan
olarak kapalıdır. Kullanım ve kod haritası:
[training protocol](docs/training-protocol.md#9-implementation-and-first-colab-run).

## Klasörler

```text
src/turkish_punctuation/  Ortak Python kodu
scripts/                 Çalıştırılabilir yardımcı dosyalar
notebooks/               İnceleme ve Colab notebook'ları
configs/                 Tokenizer, hazırlama ve deney ayarları
data/raw/                Yerel orijinal veri (Git dışında)
data/processed/          İşlenmiş veri (Git dışında)
artifacts/               Model ve deney çıktıları (Git dışında)
docs/                    Proje kararları ve araştırma notları
requirements/            Doğrulanmış yerel ortamın paket sürümleri
```

## Git ile çalışma

Küçük, anlamlı adımları commit et; veri ve ağırlıkları repoya koyma.
Notebook commit etmeden önce **Clear All Outputs** yapıp kaydet.

```bash
git status
git diff
git add <degistirdigin-dosyalar>
git commit -m "Describe the change"
git push
```

Deneysel veya büyük bir değişikliğe başlarken:

```bash
git switch -c feature/short-description
```

İsteğe bağlı kod kontrolü ve biçimlendirme:

```bash
ruff check .
ruff format .
```

Veri okuyucu, hizalama, hazırlama ve inference akışının çevrimdışı testleri:

```bash
python -m unittest discover -s tests -v
```

Testler küçük yerel tokenizer ve rastgele ağırlıklı küçük BERT fixture'ları kullanır;
veri/model indirme veya GPU gerektirmez. Bunlar araştırma deneyleri değildir.
Tam dosya ve çıktı kontrolünün bulguları: [kod incelemesi](docs/code-review.md).

## Tekrarlanabilirlik

`pyproject.toml` desteklenen paket aralıklarını belirtir.
`requirements/local-cpu.lock.txt` ilk doğrulanan yerel Python 3.11 CPU ortamındaki
paket sürümlerini kaydeder; Colab için doğrudan GPU ortam kilidi değildir.
Yerel ortamı aynı sürümlerle yeniden kurmak için Conda ortamını oluşturduktan sonra:

```bash
python -m pip install -r requirements/local-cpu.lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install -e . --no-deps
```

Deneyler başladığında veri sürümü, seed, ayarlar, Git commit'i ve sonuçlar
birlikte kaydedilecek.

2026-10-17 yerel kurulum kontrolü: Python 3.11.17, PyTorch 2.14.1+cpu,
Transformers 4.57.6. Paket uyumluluk kontrolü, BERT/ELECTRA/ConvBERT sınıflarının
yüklenmesi ve başlangıç notebook'unun gerçek kernel ile çalıştırılması başarılı.
Bu kontroller model eğitimi veya deney sonucu değildir.

## Araştırma kapsamı

Üç kıyas noktası: yayımlanmış hazır model, temiz veriyle fine-tune baseline ve
aynı mimarinin sentetik noise ile artırılmış veriyle fine-tune sürümü.
Araştırma soruları ve alınmış kararlar için [proje notları](docs/project-brief.md).

Başlangıç kaynağı: [Kurt & Çayır (2023) resmi deposu](https://github.com/uygarkurt/Turkish-Punctuation-Restoration).
İndirme, sınıf dağılımı, boş token'lar ve tekrarlarla ilgili bulgular:
[ilk veri incelemesi](docs/data-audit.md). Model lisansı ile veri lisansı ayrı ele alınır.
