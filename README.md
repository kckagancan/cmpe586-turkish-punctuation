# CMPE 586 — Turkish Punctuation Restoration

**Robustness of Turkish Punctuation Restoration under Informal Writing Noise**

CMPE 586 Transformer/NLP ders projesi. Türkçe noktalama geri yükleme modellerinin
büyük/küçük harf kaybı, Türkçe karakterlerin sadeleştirilmesi ve yazım hataları
altındaki dayanıklılığı incelenecek.

**Durum:** Yerel ortam ve public repo hazır. Aday veri seti yerelde indirildi ve
ilk yapısal inceleme tamamlandı. Veri lisansı ve kesin kaynak/split geçmişi henüz
doğrulanmadı. Henüz deney yapılmadı.

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
Python ve Jupyter VS Code eklentileri önerilen eklentiler arasında tanımlı.

`notebooks/00_environment_check.ipynb` ilk ortam kontrolü içindir.
`notebooks/01_data_inspection.ipynb` veri setinin tüm bölümlerini okur;
yapı kontrollerini, etiket dağılımlarını ve boş token'ları gösterir.
Ortak okuyucu ve doğrulama kodu `src/turkish_punctuation/data.py` içindedir;
notebook ve genel inceleme aracı aynı kodu kullanır.
Tarayıcı tabanlı Jupyter tercih edilirse `jupyter lab` çalıştırılabilir.

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
başlatmak gerekebilir. Colab uyumluluğu ve eğitim henüz denenmedi.
Uzun eğitimlerde çıktıları Drive gibi kalıcı depolamaya kaydetmeyi ayarlayacağız.

## Klasörler

```text
src/turkish_punctuation/  Ortak Python kodu
scripts/                 Çalıştırılabilir yardımcı dosyalar
notebooks/               İnceleme ve Colab notebook'ları
configs/                 Daha sonra eklenecek deney ayarları
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

Veri okuyucunun otomatik kontrolleri:

```bash
python -m unittest discover -s tests -v
```

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
