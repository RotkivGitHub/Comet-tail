import sys
import json
import math
import time
import shutil
import hashlib
import logging
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import plotly
import plotly.graph_objects as go
from pyhdf.SD import SD, SDC
from astroquery.jplhorizons import Horizons
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord, HeliocentricMeanEcliptic
from sunpy.coordinates import frames as sunpy_frames

# ==========================================
# НАЛАШТУВАННЯ
# ==========================================
# Каталог розв'язків на сайті. Розв'язок - ОДИН ФАЙЛ НА ОБЕРТ КЕРРІНГТОНА (~27,3 доби):
#   {DATA_BASE_URL}/cr<номер оберту>-<роздільність>/<модель>/<helio|corona>/br002.hdf
DATA_BASE_URL = "https://www.predsci.com/data/runs"

# Папка, куди зберігати файли
DEST_FOLDER = Path(r"C:\Users\gusar\OneDrive\Desktop\Vitishka data from MHD")

# Модель = значення #ps-data-id на сайті: "<роздільність>/<модель>".
# ВАЖЛИВО: модель має ПОКРИВАТИ вибрані дати. Для 2007 року на сайті є лише MDI-моделі
# (HMI працює з 2010 р., тож HMI-розв'язків за 2007 рік не існує).
MODEL_ID = "high/hmi_mast_mas_std_0201"
MODEL_NAME = "hmi__high-cor-thermo2-exp01__high-hel-poly-exp01"
MODEL_TAG = "thermo2_hel-poly"

# Розв'язок: "helio" - геліосферний, "corona" - корональний
SOLUTION = "helio"

# --- Період дат (UT) ---
# Формат: "YYYY-MM-DD HH:MM" (або "YYYY-MM-DD" - тоді час 00:00).
# START == END - одна дата (звичайний статичний графік).
START = "2022-12-04"
END = "2022-12-22"      # кінець включається, якщо потрапляє на крок

# Крок між сусідніми датами, години (можна дробове: 0.5 = 30 хв)
STEP_HOURS = 48

# Як часто змінюється сфера під час відтворення, секунди
FRAME_SECONDS = 2

# Кольорова шкала Br (нТ):
#   "log"    - симетрична логарифмічна (symlog): видно і сильні, і слабкі поля
#   "linear" - лінійна, симетрична навколо нуля
# Шкала спільна для всіх дат, тож колір означає те саме значення в кожному кадрі.
COLOR_SCALE = "log"

# Що робити, якщо два кадри (різні дати) вийшли ОДНАКОВИМИ (однакові значення Br):
#   "error" - зупинитись із поясненням (за замовчуванням)
#   "skip"  - пропустити такі дати й побудувати анімацію з рештою
#   "warn"  - лише попередження, кадри залишаються
ON_DUPLICATE_DATA = "error"

# Максимум точок НА КАДР, які малюються в браузері (WebGL). Якщо вибрано більше -
# показується випадкова підвибірка (однакові точки в усіх кадрах). У HTML
# зберігаються всі кадри, тому розмір ~ точки x кількість дат x 27 байт.
MAX_PLOT_POINTS = 500_000

# Якщо файл оберту вже завантажено - не завантажувати повторно
REUSE_EXISTING = True

# --- Параметри візуалізації ---
# Радіальний діапазон, R_sun
R_MIN = 30.0
R_MAX = 40.0

# Довгота сектора: як вона задається
#   "earth" - сектор СЛІДУЄ за обертанням Сонця: центр = довгота центрального меридіана
#             (напрямок на Землю, L0) на кожну дату. Файл розв'язку один на оберт, тому
#             різні дати одного оберту відрізняються САМЕ цим (Сонце повертається ~13,2°/добу).
#             LONGITUDE_CENTER_DEG тут - зсув від напрямку на Землю (0 = дивимось на Землю).
#   "fixed" - сектор нерухомий у системі Керрінгтона (LONGITUDE_CENTER_DEG - довгота
#             Керрінгтона). Усі дати одного оберту дадуть ОДНАКОВІ кадри.
LONGITUDE_MODE = "fixed"
LONGITUDE_CENTER_DEG = 0.0
LONGITUDE_HALF_WIDTH_DEG = 90.0

# Полярний кут (колатитуда, 0° = північний полюс, 90° = екватор), градуси
POLAR_ANGLE_MIN_DEG = 0.0
POLAR_ANGLE_MAX_DEG = 180.0

# Перехід від кодових одиниць MAS (так записано Br у HDF) до нанотесла.
# B0 = V0*sqrt(4*pi*rho0) = 2.20689 Гс (нормування MAS, psi_io.units.FN_B);
# 1 Гс = 1e5 нТ  ->  1 кодова одиниця = 2.20689e5 нТ
BR_CODE_TO_NT = 2.2068914e5

# Параметри логарифмічної шкали (діють, коли COLOR_SCALE = "log").
# колір ~ sign(Br) * log10(1 + |Br| / SYMLOG_LINTHRESH_NT); |Br| нижче порога - лінійно.
# None - поріг підбирається автоматично (степінь 10 нижче 5-го перцентиля |Br|).
# Менше значення -> слабкі поля яскравіші.
SYMLOG_LINTHRESH_NT = None

# Вигляд маркерів
MARKER_SIZE = 4
MARKER_OPACITY = 0.9

# Режим графіка:
#   "spherical" - справжня сферична геометрія (X, Y, Z у R☉) + сітка меридіанів
#                 і паралелей з підписами довготи та полярного кута в градусах
#   "angles"    - осі: довгота (°), полярний кут (°), радіус (R☉)
PLOT_MODE = "spherical"

# Комета для нанесення на кожен кадр. Наприклад: "2021 A1" (C/2021 A1 Leonard).
# Порожній рядок вимикає запит JPL Horizons і маркер комети.
COMET_NAME = "2021 A1"
COMET_MARKER_SIZE = 8
COMET_MARKER_COLOR = "black"
# Радіальний промінь від центра Сонця закінчується тут: R_MAX + цей запас.
COMET_RAY_MARGIN_RSUN = 20.0

# Кількість точок на кожній лінії сітки (для плавності дуг)
GRATICULE_POINTS = 80

# ==========================================
# ЛОГУВАННЯ
# ==========================================
DEST_FOLDER.mkdir(parents=True, exist_ok=True)
LOG_DIR = DEST_FOLDER / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"run_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log"

log = logging.getLogger("mhdweb")
log.setLevel(logging.DEBUG)
log.handlers.clear()
log.propagate = False
_fmt = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(funcName)-24s | %(message)s",
    datefmt="%H:%M:%S",
)
_console = logging.StreamHandler(sys.stdout)
_console.setLevel(logging.INFO)
_console.setFormatter(_fmt)
_file = logging.FileHandler(LOG_FILE, encoding="utf-8")
_file.setLevel(logging.DEBUG)
_file.setFormatter(_fmt)
log.addHandler(_console)
log.addHandler(_file)


def mb(nbytes):
    return nbytes / 1024 ** 2


def log_array(name, arr):
    """Статистика числового масиву."""
    arr = np.asarray(arr)
    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    log.info("%s: shape=%s, dtype=%s, %.2f МБ, NaN=%d (%.3f%%), Inf=%d",
             name, arr.shape, arr.dtype, mb(arr.nbytes),
             n_nan, 100.0 * n_nan / max(arr.size, 1), n_inf)
    finite = arr[np.isfinite(arr)]
    if finite.size:
        log.info("%s: min=%.6g, max=%.6g, mean=%.6g, std=%.6g",
                 name, finite.min(), finite.max(), finite.mean(), finite.std())


def log_coord(name, arr, unit=""):
    arr = np.asarray(arr)
    log.info("%s: N=%d, діапазон=[%.6g, %.6g] %s",
             name, len(arr), arr.min(), arr.max(), unit)


class DataMismatchError(RuntimeError):
    """Потрібних даних немає на сайті або файл не той, що очікувався."""


class DuplicateDataError(RuntimeError):
    """Для різних дат вийшли однакові кадри."""


# ==========================================
# ОБЕРТ КЕРРІНГТОНА ТА ПОСИЛАННЯ НА ФАЙЛ
# ==========================================
def julian_date(dt):
    """Юліанська дата для datetime в UT."""
    return 2440587.5 + (dt - datetime(1970, 1, 1)).total_seconds() / 86400.0


def carrington(dt):
    """
    Номер оберту Керрінгтона та довгота центрального меридіана L0 (напрямок на Землю),
    градуси, для дати dt (UT). Середнє обертання; та сама конвенція, що на MHDweb
    (напр. 2007-09-25 12:00 -> CR 2061, L0 ~ 168°).
    """
    x = (julian_date(dt) - 2398140.2270) / 27.2752316
    cr = int(math.floor(x))
    l0 = (360.0 * (1.0 - (x - cr))) % 360.0
    return cr, l0


def split_model():
    if not MODEL_ID or "/" not in MODEL_ID:
        raise ValueError("MODEL_ID має вигляд '<роздільність>/<модель>', "
                         "напр. 'medium/mdi_mas_mas_std_0101'")
    res, model = MODEL_ID.split("/", 1)
    return res, model


def hdf_url(cr):
    """Пряме посилання на br002.hdf для оберту cr вибраної моделі."""
    res, model = split_model()
    return f"{DATA_BASE_URL}/cr{cr}-{res}/{model}/{SOLUTION}/br002.hdf"


def hdf_cache_path(cr):
    res, model = split_model()
    return DEST_FOLDER / f"br002_cr{cr}_{res}_{model}_{SOLUTION}.hdf"


HDF4_MAGIC = b"\x0e\x03\x13\x01"


def is_hdf4(path):
    try:
        with open(path, "rb") as f:
            return f.read(4) == HDF4_MAGIC
    except OSError:
        return False


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def reject_file(path):
    """Перейменовує непридатний файл на *.rejected (нічого не видаляється)."""
    path = Path(path)
    target, n = path.with_name(path.name + ".rejected"), 1
    while target.exists():
        target = path.with_name(f"{path.name}.rejected{n}")
        n += 1
    path.rename(target)
    log.warning("Файл не пройшов перевірку: %s -> %s", path.name, target.name)


def download_direct(url, dest):
    """Пряме HTTP-завантаження. 404/410 -> DataMismatchError (файлу немає на сайті)."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) mhdweb-br-viewer"}
    tmp = dest.with_name(dest.name + ".part")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                    timeout=120) as resp, open(tmp, "wb") as f:
            log.info("HTTP %s, Content-Type: %s, Content-Length: %s",
                     getattr(resp, "status", "?"),
                     resp.headers.get("Content-Type"), resp.headers.get("Content-Length"))
            shutil.copyfileobj(resp, f, 1 << 20)
    except urllib.error.HTTPError as e:
        tmp.unlink(missing_ok=True)
        if e.code in (404, 410):
            raise DataMismatchError(
                f"файлу немає на сайті (HTTP {e.code}): {url}. Імовірно, вибрана модель "
                f"«{MODEL_ID}» не має розв'язку для цього оберту Керрінгтона") from e
        raise
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(dest)
    log.info("Завантажено за %.2f с", time.perf_counter() - t0)


def download_via_browser(url, dest):
    """Запасний шлях: те саме посилання, але через браузер (без роботи з формою сайту)."""
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page(accept_downloads=True)
            with page.expect_download(timeout=120_000) as info:
                try:
                    page.goto(url)
                except Exception:
                    pass  # перехід перетворюється на завантаження - це нормально
            info.value.save_as(str(dest))
        finally:
            browser.close()


def ensure_hdf(cr):
    """
    Повертає шлях до HDF оберту cr. Файл кешується ОДИН РАЗ НА ОБЕРТ
    (br002_cr<N>_<роздільність>_<модель>_<розв'язок>.hdf) і використовується всіма датами цього оберту.
    """
    url, dest = hdf_url(cr), hdf_cache_path(cr)
    if REUSE_EXISTING and dest.exists() and dest.stat().st_size > 0:
        if is_hdf4(dest):
            log.info("Файл оберту CR %d уже є: %s", cr, dest.name)
            return str(dest)
        log.warning("Наявний файл %s не схожий на HDF - завантажую знову", dest.name)
        reject_file(dest)

    log.info("Завантаження CR %d: %s", cr, url)
    try:
        download_direct(url, dest)
    except DataMismatchError:
        raise
    except Exception as e:
        log.warning("Пряме завантаження не вдалося (%s) - пробую через браузер", e)
        download_via_browser(url, dest)

    size = dest.stat().st_size if dest.exists() else 0
    if size == 0 or not is_hdf4(dest):
        if dest.exists():
            reject_file(dest)
        raise DataMismatchError(f"завантажений файл порожній або не є HDF: {url}")

    sha = sha256_file(dest)
    Path(str(dest) + ".json").write_text(json.dumps({
        "url": url, "cr": cr, "model_id": MODEL_ID, "solution": SOLUTION,
        "size": size, "sha256": sha, "downloaded": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Збережено: %s (%.2f МБ), sha256: %s", dest.name, mb(size), sha[:16])
    return str(dest)


# ==========================================
# ЗЧИТУВАННЯ HDF
# ==========================================
def load_br_hdf(file_path):
    """
    Читає HDF-файл MHDweb і повертає (phi, theta, radius, Br),
    де Br має порядок осей (phi, theta, radius), а координати відсортовані.
    phi, theta - у радіанах, radius - у R_sun.
    """
    file_path = Path(file_path)
    log.info("Читаю HDF: %s", file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Не знайдено HDF-файл:\n{file_path}")
    log.info("Розмір файлу: %.2f МБ", mb(file_path.stat().st_size))

    t0 = time.perf_counter()
    hdf = SD(str(file_path), SDC.READ)
    try:
        for name, info in hdf.datasets().items():
            log.info("  набір даних %s: %s", name, info)
        phi = np.asarray(hdf.select("fakeDim0")[:], dtype=float)
        theta = np.asarray(hdf.select("fakeDim1")[:], dtype=float)
        radius = np.asarray(hdf.select("fakeDim2")[:], dtype=float)
        Br = np.asarray(hdf.select("Data-Set-2")[:], dtype=float)
    finally:
        hdf.end()
    log.info("Дані прочитано за %.2f с", time.perf_counter() - t0)

    log_coord("phi", phi, "рад")
    log_coord("theta", theta, "рад")
    log_coord("radius", radius, "R_sun")
    log_array("Br (початковий, кодові одиниці MAS)", Br)

    # --- Перевірки структури ---
    expected_shape = (len(phi), len(theta), len(radius))
    if Br.shape != expected_shape:
        raise ValueError(
            f"Неочікувана форма Br: {Br.shape}. Очікувалася {expected_shape}. "
            "Потрібно перевірити порядок осей."
        )
    for cname, carr in [("phi", phi), ("theta", theta), ("radius", radius)]:
        if not np.all(np.isfinite(carr)):
            raise ValueError(f"Координати {cname} містять нечислові значення.")
    if np.any(np.diff(np.sort(theta)) <= 0):
        raise ValueError("Виявлено дублікати координат theta.")
    if np.any(np.diff(np.sort(radius)) <= 0):
        raise ValueError("Виявлено дублікати координат radius.")
    log.info("Перевірки структури пройдено")

    # --- Сортування координат ---
    phi_mod = np.mod(phi, 2 * np.pi)
    ip = np.argsort(phi_mod)
    it = np.argsort(theta)
    ir = np.argsort(radius)
    phi = phi_mod[ip]
    theta = theta[it]
    radius = radius[ir]
    Br = Br[np.ix_(ip, it, ir)]

    # Усунення повторної довготи, якщо вона є
    n_before = len(phi)
    phi, unique_indices = np.unique(phi, return_index=True)
    Br = Br[unique_indices, :, :]
    if n_before != len(phi):
        log.warning("Видалено %d повторних вузлів за довготою", n_before - len(phi))
    if len(phi) < 2:
        raise ValueError("Недостатньо вузлів за довготою.")

    log.info("Після сортування: Br shape=%s", Br.shape)
    return phi, theta, radius, Br


# ==========================================
# ВИБІР СЕКТОРА ДЛЯ ОДНІЄЇ ДАТИ
# ==========================================
def make_sector(phi, theta, radius, Br, l0, verbose=True):
    """
    Відбирає вузли сектора (R, довгота, полярний кут) для дати з довготою центрального меридіана l0.
    Повертає словник: lon (довгота для осі), lon_true (довгота Керрінгтона 0-360°),
    pol, rad (геометрія) і br (Br у нТ).

    "fixed": вузли сітки в нерухомому вікні довготи (система Керрінгтона).
    "earth": вікно рухається разом із L0; Br інтерполюється вздовж довготи на ФІКСОВАНУ
             сітку відносних довгот (відносно напрямку на Землю), тож набір точок однаковий
             у всіх кадрах, а змінюється тільки Br.
    """
    say = log.info if verbose else log.debug
    pol_all = np.degrees(theta)

    if LONGITUDE_MODE == "fixed":
        PHI, THETA, R = np.meshgrid(phi, theta, radius, indexing="ij")
        LON_GRID = np.degrees(PHI)
        POLAR_GRID = np.degrees(THETA)
        lon_diff = ((LON_GRID - LONGITUDE_CENTER_DEG + 180) % 360) - 180
        LON_PLOT = LONGITUDE_CENTER_DEG + lon_diff   # неперервна довгота навколо центру

        cond_r = (R >= R_MIN) & (R <= R_MAX)
        cond_lon = np.abs(lon_diff) <= LONGITUDE_HALF_WIDTH_DEG
        cond_pol = (POLAR_GRID >= POLAR_ANGLE_MIN_DEG) & (POLAR_GRID <= POLAR_ANGLE_MAX_DEG)
        say("Доступний діапазон: R=[%.2f, %.2f] R_sun, довгота=[%.1f, %.1f]°, "
            "полярний кут=[%.1f, %.1f]°", radius.min(), radius.max(),
            LON_GRID.min(), LON_GRID.max(), POLAR_GRID.min(), POLAR_GRID.max())
        for nm, c in [("радіус", cond_r), ("довгота", cond_lon), ("полярний кут", cond_pol)]:
            say("  умова «%s»: %d вузлів (%.2f%%)", nm, int(c.sum()), 100.0 * c.mean())
        mask = cond_r & cond_lon & cond_pol
        lon, lon_true = LON_PLOT[mask], np.mod(LON_GRID[mask], 360.0)
        pol, rad = POLAR_GRID[mask], R[mask]
        br = Br[mask] * BR_CODE_TO_NT

    elif LONGITUDE_MODE == "earth":
        mt = (pol_all >= POLAR_ANGLE_MIN_DEG) & (pol_all <= POLAR_ANGLE_MAX_DEG)
        mr = (radius >= R_MIN) & (radius <= R_MAX)
        if not mt.any() or not mr.any():
            raise ValueError("У вибраній області немає вузлів. "
                             f"Доступно: R=[{radius.min():.1f}, {radius.max():.1f}] R_sun, "
                             f"полярний кут=[{pol_all.min():.1f}, {pol_all.max():.1f}]°.")
        th_s, r_s = pol_all[mt], radius[mr]
        Bsub = Br[:, mt, :][:, :, mr]

        # Фіксована сітка ВІДНОСНИХ довгот (крок = крок вихідної сітки)
        phi_deg = np.degrees(phi)
        dphi = float(np.median(np.diff(phi_deg)))
        if LONGITUDE_HALF_WIDTH_DEG >= 180:
            K = int(round(360.0 / dphi))
            rel = LONGITUDE_CENTER_DEG + (np.arange(K) - K // 2) * (360.0 / K)
        else:
            n = int(np.floor(LONGITUDE_HALF_WIDTH_DEG / dphi))
            rel = LONGITUDE_CENTER_DEG + np.arange(-n, n + 1) * dphi
        abs_lon = np.mod(l0 + rel, 360.0)       # довгота Керрінгтона цих точок на цю дату

        # Періодична лінійна інтерполяція вздовж довготи
        p_ext = np.concatenate(([phi_deg[-1] - 360.0], phi_deg, [phi_deg[0] + 360.0]))
        idx = np.clip(np.searchsorted(p_ext, abs_lon, side="right") - 1, 0, len(p_ext) - 2)
        w = (abs_lon - p_ext[idx]) / (p_ext[idx + 1] - p_ext[idx])
        B_ext = np.concatenate((Bsub[-1:], Bsub, Bsub[:1]), axis=0)
        Bi = (1.0 - w)[:, None, None] * B_ext[idx] + w[:, None, None] * B_ext[idx + 1]

        REL, TH, RR = np.meshgrid(rel, th_s, r_s, indexing="ij")
        say("Сектор слідує за Сонцем: L0=%.2f°, відносні довготи [%.1f, %.1f]° "
            "(Керрінгтон: центр %.1f°), вузлів: %d x %d x %d",
            l0, rel.min(), rel.max(), (l0 + LONGITUDE_CENTER_DEG) % 360.0,
            len(rel), len(th_s), len(r_s))
        lon = REL.ravel()
        lon_true = np.mod(l0 + REL, 360.0).ravel()
        pol, rad = TH.ravel(), RR.ravel()
        br = Bi.ravel() * BR_CODE_TO_NT
    else:
        raise ValueError("LONGITUDE_MODE має бути 'earth' або 'fixed'")

    if len(br) == 0:
        raise ValueError(
            "У вибраній області немає вузлів. "
            f"Доступно: R=[{radius.min():.1f}, {radius.max():.1f}] R_sun, "
            f"полярний кут=[{pol_all.min():.1f}, {pol_all.max():.1f}]°. "
            "Перевір межі радіуса, довготи та полярного кута."
        )

    # Однаковий набір точок у всіх кадрах: NaN/Inf не відкидаємо, а замінюємо на 0
    n_bad = int((~np.isfinite(br)).sum())
    if n_bad:
        log.warning("У вибраній області %d значень Br з NaN/Inf - замінено на 0", n_bad)
        br = np.nan_to_num(br, nan=0.0, posinf=0.0, neginf=0.0)

    say("Кількість вузлів сектора: %d", len(br))
    log_array("Br (вибране, нТ)", br)
    return dict(lon=lon, lon_true=lon_true, pol=pol, rad=rad, br=br, l0=l0)


# ==========================================
# ПОЛОЖЕННЯ КОМЕТИ З JPL HORIZONS
# ==========================================
def get_comet_position(dt, l0):
    """Запитує координати Horizons, переводить їх у Carrington та координати графіка."""
    t = Time(dt, scale="utc")
    query_id = f"DES={COMET_NAME.strip()};CAP"
    obj = Horizons(id=query_id, id_type="smallbody", location="500@10", epochs=[float(t.jd)])
    eph = obj.ephemerides(quantities="18,19")
    if len(eph) == 0:
        raise RuntimeError(f"Horizons не повернув епімерид для {COMET_NAME} на {dt}.")

    coord_ecl = SkyCoord(
        lon=float(eph["EclLon"][0]) * u.deg,
        lat=float(eph["EclLat"][0]) * u.deg,
        distance=float(eph["r"][0]) * u.au,
        frame=HeliocentricMeanEcliptic(equinox=Time("J2000"), obstime=t),
    )
    carr = coord_ecl.transform_to(
        sunpy_frames.HeliographicCarrington(obstime=t, observer="earth")
    )
    carr_lon = float(carr.lon.to_value(u.deg)) % 360.0
    heliographic_lat = float(carr.lat.to_value(u.deg))
    radius_rsun = float(carr.radius.to_value(u.km) / 695700.0)
    polar_theta = 90.0 - heliographic_lat

    if LONGITUDE_MODE == "earth":
        plot_lon = LONGITUDE_CENTER_DEG + ((carr_lon - l0 + 180.0) % 360.0 - 180.0)
    else:
        plot_lon = LONGITUDE_CENTER_DEG + ((carr_lon - LONGITUDE_CENTER_DEG + 180.0) % 360.0 - 180.0)

    if PLOT_MODE == "spherical":
        theta_r, lon_r = math.radians(polar_theta), math.radians(plot_lon)
        x = radius_rsun * math.sin(theta_r) * math.cos(lon_r)
        y = radius_rsun * math.sin(theta_r) * math.sin(lon_r)
        z = radius_rsun * math.cos(theta_r)
    else:
        x, y, z = plot_lon, polar_theta, radius_rsun
    return {
        "x": x,
        "y": y,
        "z": z,
        "carr_lon": carr_lon,
        "lat": heliographic_lat,
        "theta": polar_theta,
        "radius": radius_rsun,
        "plot_lon": plot_lon,
    }


# ==========================================
# ВІЗУАЛІЗАЦІЯ
# ==========================================
def nice_degree_ticks(lo, hi, wrap=False, signed=False, max_ticks=8):
    """
    Підбирає «круглі» поділки осі в градусах.
    wrap=True - підписи приводяться до 0-360° (вісь при цьому лишається неперервною).
    signed=True - підписи зі знаком (для відносних довгот).
    """
    span = max(hi - lo, 1e-9)
    step = next((s for s in (1, 2, 5, 10, 15, 20, 30, 45, 60, 90)
                 if span / s <= max_ticks), 90)
    vals = np.arange(np.ceil(lo / step) * step, hi + 1e-9, step)

    def fmt(v):
        if wrap:
            v = v % 360
        if signed:
            return "0°" if abs(v) < 1e-9 else f"{v:+g}°"
        return f"{v:g}°"
    return vals.tolist(), [fmt(v) for v in vals]


def spherical_graticule(lon_vals, lon_texts, pol_vals, pol_texts,
                        lon_lo, lon_hi, pol_lo, pol_hi, r_shell, lon_prefix="φ="):
    """
    Будує сферичну сітку на оболонці радіуса r_shell:
    меридіани (довгота) і паралелі (полярний кут) з підписами в градусах.
    Повертає (trace_ліній, trace_підписів).
    """
    def to_xyz(r, th_deg, lon_deg):
        th = np.radians(th_deg)
        ph = np.radians(lon_deg)
        return r * np.sin(th) * np.cos(ph), r * np.sin(th) * np.sin(ph), r * np.cos(th)

    xs, ys, zs = [], [], []
    lx, ly, lz, lt = [], [], [], []
    r_lab = r_shell * 1.03  # підписи трохи поза оболонкою

    # Меридіани: лінія вздовж полярного кута при фіксованій довготі
    th_line = np.linspace(pol_lo, pol_hi, GRATICULE_POINTS)
    th_label = float(np.clip(90.0, pol_lo, pol_hi))  # підпис ближче до екватора
    seen = set()
    for v, t in zip(lon_vals, lon_texts):
        key = round(v % 360, 6)
        if key in seen:   # напр. -180° і +180° - це той самий меридіан
            continue
        seen.add(key)
        x, y, z = to_xyz(r_shell, th_line, np.full_like(th_line, v))
        xs += x.tolist() + [None]
        ys += y.tolist() + [None]
        zs += z.tolist() + [None]
        a, b, c = to_xyz(r_lab, th_label, v)
        lx.append(float(a)); ly.append(float(b)); lz.append(float(c))
        lt.append(f"{lon_prefix}{t}")

    # Паралелі: лінія вздовж довготи при фіксованому полярному куті
    lon_line = np.linspace(lon_lo, lon_hi, GRATICULE_POINTS)
    for v, t in zip(pol_vals, pol_texts):
        if abs(np.sin(np.radians(v))) < 1e-6:   # полюс - це точка, не лінія
            continue
        x, y, z = to_xyz(r_shell, np.full_like(lon_line, v), lon_line)
        xs += x.tolist() + [None]
        ys += y.tolist() + [None]
        zs += z.tolist() + [None]
        a, b, c = to_xyz(r_lab, v, lon_lo)
        lx.append(float(a)); ly.append(float(b)); lz.append(float(c))
        lt.append(f"θ={t}")

    lines = go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines",
        line=dict(color="gray", width=2),
        hoverinfo="skip", showlegend=False, name="Сітка φ/θ",
    )
    labels = go.Scatter3d(
        x=lx, y=ly, z=lz, mode="text", text=lt,
        textfont=dict(size=11, color="dimgray"),
        hoverinfo="skip", showlegend=False, name="Підписи φ/θ",
    )
    return lines, labels


def symlog(x, lin):
    """Симетричний логарифм: sign(x) * log10(1 + |x|/lin)."""
    return np.sign(x) * np.log10(1.0 + np.abs(x) / lin)


def log_scale_params(br_list):
    """
    Поріг лінійності symlog і максимум |Br| за ВСІМА датами разом, щоб колір
    означав те саме значення в кожному кадрі (кадри можна порівнювати).
    """
    abs_all = np.concatenate([np.abs(np.asarray(b, dtype=float)).ravel() for b in br_list])
    nz = abs_all[abs_all > 0]
    if SYMLOG_LINTHRESH_NT:
        lin = float(SYMLOG_LINTHRESH_NT)
    elif nz.size:
        lin = 10.0 ** np.floor(np.log10(max(float(np.percentile(nz, 5)), 1e-9)))
    else:
        lin = 1.0
    br_max = float(np.nanmax(abs_all)) if abs_all.size else 1.0
    return lin, br_max


def log_colorbar(lin, br_max):
    """Повертає (cmax, налаштування_colorbar); підписи - справжні значення Br у нТ."""
    cmax = float(symlog(br_max, lin))
    if not np.isfinite(cmax) or cmax == 0:
        cmax = 1.0

    # Поділки: 1, 3, 10, 30, ... нТ (від порога до максимуму), для обох знаків
    ticks = sorted({round(m * 10.0 ** k, 12) for k in range(-9, 10) for m in (1, 3)
                    if lin * 0.999 <= m * 10.0 ** k <= br_max * 1.001})
    tickvals = ([-float(symlog(t, lin)) for t in reversed(ticks)] + [0.0]
                + [float(symlog(t, lin)) for t in ticks])
    ticktext = ([f"-{t:g}" for t in reversed(ticks)] + ["0"] + [f"{t:g}" for t in ticks])
    cfg = dict(title="Br, нТ<br>(лог. шкала)", tickmode="array",
               tickvals=tickvals, ticktext=ticktext)
    log.info("Логарифмічна шкала: поріг лінійності = %.3g нТ, |Br| max = %.4g нТ, "
             "поділки: %s нТ", lin, br_max, ticks)
    return cmax, cfg


def linear_colorbar(br_list):
    """Повертає (cmax, налаштування_colorbar) для лінійної симетричної шкали."""
    cmax = max(float(np.nanmax(np.abs(np.asarray(b, dtype=float)))) for b in br_list)
    if not np.isfinite(cmax) or cmax == 0:
        cmax = 1.0
    log.info("Лінійна шкала: [%.4g, %.4g] нТ", -cmax, cmax)
    return cmax, dict(title="Br, нТ")


def ranges_for_axes(sector):
    """Діапазони вибраних точок (для осей і сітки)."""
    return (float(sector["lon"].min()), float(sector["lon"].max()),
            float(sector["pol"].min()), float(sector["pol"].max()),
            float(sector["rad"].min()), float(sector["rad"].max()))


# JS для відтворення: автоматична зміна кадрів кожні __MS__ мс, пауза, крок вперед/назад.
# {plot_id} підставляє plotly.
PLAYER_JS = """
(function () {
  var gd = document.getElementById('{plot_id}');
  var names = __NAMES__;
  var ms = __MS__;
  var idx = 0, timer = null;

  var box = document.createElement('div');
  box.style.cssText = 'position:fixed;top:8px;left:8px;z-index:1000;font:14px sans-serif;' +
    'background:rgba(255,255,255,0.92);padding:6px 10px;border-radius:6px;border:1px solid #ccc;';
  box.innerHTML = '<button id="pb" style="margin-right:6px"></button>' +
    '<button id="pv">&#9664;</button> <button id="nx">&#9654;</button>' +
    '<span style="margin-left:8px;color:#444">зміна кожні ' + (ms / 1000) + ' с</span>';
  document.body.appendChild(box);
  var pb = document.getElementById('pb');

  function show(i) {
    idx = (i + names.length) % names.length;
    Plotly.animate(gd, [names[idx]], {
      mode: 'immediate', frame: {duration: 0, redraw: true}, transition: {duration: 0}
    });
  }
  function play() {
    if (timer) return;
    timer = setInterval(function () { show(idx + 1); }, ms);
    pb.textContent = '\\u23F8 Пауза';
  }
  function pause() {
    if (timer) { clearInterval(timer); timer = null; }
    pb.textContent = '\\u25B6 Відтворити';
  }
  pb.onclick = function () { if (timer) { pause(); } else { play(); } };
  document.getElementById('pv').onclick = function () { pause(); show(idx - 1); };
  document.getElementById('nx').onclick = function () { pause(); show(idx + 1); };
  gd.on('plotly_sliderchange', function (e) {
    var n = (e.step && e.step.args) ? e.step.args[0][0] : null;
    var k = names.indexOf(n);
    if (k >= 0) idx = k;
    if (e.interaction) pause();
  });
  play();
})();
"""


def visualize_sectors(sectors, labels, dates, comet_positions):
    """
    Будує 3D-графік. Для однієї дати - звичайний графік; для кількох - анімацію:
    кадр на дату, зміна кожні FRAME_SECONDS секунд, кнопки паузи/кроку і повзунок.
    Колір - одна спільна шкала для всіх дат.
    """
    base = sectors[0]
    n_all = len(base["rad"])
    n_frames = len(sectors)
    earth = LONGITUDE_MODE == "earth"

    # Діапазони осей/сітки - за повними даними (до підвибірки)
    lon_lo, lon_hi, pol_lo, pol_hi, r_lo, r_hi = ranges_for_axes(base)

    # Підвибірка (однакові точки в усіх кадрах)
    if n_all > MAX_PLOT_POINTS:
        idx = np.sort(np.random.default_rng(0).choice(n_all, MAX_PLOT_POINTS, replace=False))
        log.warning("Точок у секторі %d > MAX_PLOT_POINTS=%d: показано випадкову підвибірку "
                    "(%.0f%%). Щоб бачити всі - зменш область (R_MAX, кути).",
                    n_all, MAX_PLOT_POINTS, 100.0 * MAX_PLOT_POINTS / n_all)
        sel = lambda a: a[idx]
    else:
        sel = lambda a: a
    lon, pol, rad = (sel(base[k]) for k in ("lon", "pol", "rad"))
    lon_true_list = [sel(sc["lon_true"]) for sc in sectors]
    br_list = [sel(sc["br"]) for sc in sectors]
    n_show = len(rad)

    est_mb = n_show * (n_frames * 27 + 43) / 1024 ** 2 + 5   # кадри + базовий графік + plotly.js
    log.info("Кадрів: %d; точок на кадр: %d; оцінка розміру HTML: ~%.0f МБ",
             n_frames, n_show, est_mb)
    if est_mb > 300:
        log.warning("Очікуваний HTML дуже великий (~%.0f МБ) - браузер може не відобразити "
                    "графік. Зменш MAX_PLOT_POINTS, область або кількість дат.", est_mb)

    # Спільна кольорова шкала за всіма датами (за повними даними)
    all_br = [sc["br"] for sc in sectors]
    if COLOR_SCALE == "log":
        lin, br_max = log_scale_params(all_br)
        cmax, colorbar_cfg = log_colorbar(lin, br_max)
        color_list = [symlog(b, lin).astype(np.float32) for b in br_list]
    elif COLOR_SCALE == "linear":
        cmax, colorbar_cfg = linear_colorbar(all_br)
        color_list = [np.asarray(b, dtype=np.float32) for b in br_list]
    else:
        raise ValueError("COLOR_SCALE має бути 'log' або 'linear'")

    # Координати точок. lon - довгота для осі (у режимі "earth" - відносно напрямку на Землю)
    if PLOT_MODE == "spherical":
        theta_r = np.radians(pol)
        lon_r = np.radians(lon)
        px = rad * np.sin(theta_r) * np.cos(lon_r)
        py = rad * np.sin(theta_r) * np.sin(lon_r)
        pz = rad * np.cos(theta_r)
    elif PLOT_MODE == "angles":
        px, py, pz = lon, pol, rad
    else:
        raise ValueError("PLOT_MODE має бути 'spherical' або 'angles'")
    log.info("Режим графіка: %s, довгота: %s", PLOT_MODE, LONGITUDE_MODE)
    px, py, pz = (np.asarray(a, dtype=np.float32) for a in (px, py, pz))

    def hover_data(i):
        # Підказка формується в браузері з customdata (замість рядка на кожну точку)
        return np.column_stack([lon_true_list[i] % 360, pol, rad, br_list[i]]).astype(np.float32)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=px, y=py, z=pz,
        mode="markers",
        marker=dict(
            size=MARKER_SIZE,
            color=color_list[0],   # для 'log' - symlog(Br); підказка завжди показує справжній Br
            colorscale="RdBu",
            reversescale=True,     # синій = від'ємні, червоний = додатні
            cmin=-cmax,
            cmax=cmax,
            colorbar=colorbar_cfg,
            opacity=MARKER_OPACITY
        ),
        customdata=hover_data(0),
        hovertemplate=(
            "φ (довгота Керрінгтона) = %{customdata[0]:.3f}°<br>"
            "θ (полярний кут) = %{customdata[1]:.3f}°<br>"
            "r = %{customdata[2]:.3f} R☉<br>"
            "Br = %{customdata[3]:.6g} нТ<extra></extra>"
        ),
        name="Вузли MHDweb"
    ))

    def comet_trace_data(position):
        if position is None:
            return go.Scatter3d(x=[], y=[], z=[], mode="markers", name=f"Комета {COMET_NAME}")
        return go.Scatter3d(
            x=[position["x"]], y=[position["y"]], z=[position["z"]], mode="markers",
            marker=dict(size=COMET_MARKER_SIZE, color=COMET_MARKER_COLOR,
                        line=dict(color="white", width=1)),
            customdata=[[position["carr_lon"], position["lat"],
                         position["theta"], position["radius"]]],
            hovertemplate=(f"Комета {COMET_NAME}<br>"
                           "Carrington lon = %{customdata[0]:.3f}°<br>"
                           "Геліографічна lat = %{customdata[1]:.3f}°<br>"
                           "θ (полярний кут) = %{customdata[2]:.3f}°<br>"
                           "r = %{customdata[3]:.3f} R☉<extra></extra>"),
            name=f"Комета {COMET_NAME}",
        )

    fig.add_trace(comet_trace_data(comet_positions[0]))

    def comet_ray_data(position):
        """Радіальна лінія від центра Сонця до R_MAX + запасу, у напрямку комети."""
        if position is None:
            return go.Scatter3d(x=[], y=[], z=[], mode="lines", name="Промінь до комети")
        end_r = R_MAX + COMET_RAY_MARGIN_RSUN
        if PLOT_MODE == "spherical":
            theta_r = math.radians(position["theta"])
            lon_r = math.radians(position["plot_lon"])
            end_x = end_r * math.sin(theta_r) * math.cos(lon_r)
            end_y = end_r * math.sin(theta_r) * math.sin(lon_r)
            end_z = end_r * math.cos(theta_r)
            x_values, y_values, z_values = [0.0, end_x], [0.0, end_y], [0.0, end_z]
        else:
            x_values = [position["plot_lon"], position["plot_lon"]]
            y_values = [position["theta"], position["theta"]]
            z_values = [0.0, end_r]
        return go.Scatter3d(
            x=x_values, y=y_values, z=z_values, mode="lines",
            line=dict(color="black", width=3, dash="dash"),
            hovertemplate=(f"Радіальний напрямок комети {COMET_NAME}<br>"
                           f"Кінець: {end_r:.1f} R☉<extra></extra>"),
            name="Промінь до комети",
        )

    fig.add_trace(comet_ray_data(comet_positions[0]))

    # Діапазони та поділки у градусах
    log.info("Діапазони: довгота [%.2f, %.2f]°, полярний кут [%.2f, %.2f]°, "
             "r [%.2f, %.2f] R☉", lon_lo, lon_hi, pol_lo, pol_hi, r_lo, r_hi)
    lon_tickvals, lon_ticktext = nice_degree_ticks(lon_lo, lon_hi, wrap=not earth, signed=earth)
    pol_tickvals, pol_ticktext = nice_degree_ticks(pol_lo, pol_hi)

    if PLOT_MODE == "spherical":
        # Сферична сітка на зовнішній оболонці вибраного шару
        grat_lines, grat_labels = spherical_graticule(
            lon_tickvals, lon_ticktext, pol_tickvals, pol_ticktext,
            lon_lo, lon_hi, pol_lo, pol_hi, r_hi,
            lon_prefix="Δφ=" if earth else "φ=",
        )
        fig.add_trace(grat_lines)
        fig.add_trace(grat_labels)
        log.info("Сітка: %d меридіанів, %d паралелей на r=%.2f R☉",
                 len(lon_tickvals), len(pol_tickvals), r_hi)
        scene_cfg = dict(
            xaxis_title="X (R☉)",
            yaxis_title="Y (R☉)",
            zaxis_title="Z (R☉)",
            aspectmode="data",
        )
    else:
        # Одиниці різні (градуси та R☉), тому aspectmode="data" не підходить:
        # пропорції задаємо за розмахом кутів, висота (радіус) - фіксована частка.
        lon_span = max(lon_hi - lon_lo, 1e-9)
        pol_span = max(pol_hi - pol_lo, 1e-9)
        big = max(lon_span, pol_span)
        scene_cfg = dict(
            xaxis=dict(title="Довгота відносно L0 (°)" if earth else "Довгота φ (°)",
                       tickvals=lon_tickvals, ticktext=lon_ticktext),
            yaxis=dict(title="Полярний кут θ (°)",
                       tickvals=pol_tickvals, ticktext=pol_ticktext),
            zaxis=dict(title="Радіус r (R☉)"),
            aspectmode="manual",
            aspectratio=dict(
                x=max(lon_span / big, 0.1),
                y=max(pol_span / big, 0.1),
                z=0.6,
            ),
        )

    # Заголовок
    if n_frames == 1:
        when = f"{labels[0]} UT"
    else:
        when = f"{dates[0]:%Y-%m-%d %H:%M} … {dates[-1]:%Y-%m-%d %H:%M} UT, крок {STEP_HOURS:g} год"
    if earth:
        lon_txt = (f"Δφ від L0 (напрямок на Землю) = {LONGITUDE_CENTER_DEG:g}° ± "
                   f"{LONGITUDE_HALF_WIDTH_DEG:g}°")
    else:
        lon_txt = (f"φ (Керрінгтон) = {LONGITUDE_CENTER_DEG:g}° ± "
                   f"{LONGITUDE_HALF_WIDTH_DEG:g}°")
    layout = dict(
        title=(
            f"Радіальна компонента Br | {MODEL_NAME}"
            + (f" | комета {COMET_NAME} (чорна точка)" if COMET_NAME.strip() else "")
            + "<br>"
            f"{when} | R = {R_MIN}–{R_MAX} R☉ | {lon_txt} | "
            f"θ = {POLAR_ANGLE_MIN_DEG}°–{POLAR_ANGLE_MAX_DEG}°"
        ),
        scene=scene_cfg,
        margin=dict(l=0, r=0, t=100, b=0),
    )

    post_script = None
    if n_frames > 1:
        # Кадри змінюють лише колір і підказку (координати точок однакові)
        fig.frames = [
            go.Frame(
                name=labels[i], traces=[0, 1, 2],
                data=[go.Scatter3d(marker=dict(color=color_list[i]),
                                   customdata=hover_data(i)),
                      comet_trace_data(comet_positions[i]),
                      comet_ray_data(comet_positions[i])],
            )
            for i in range(n_frames)
        ]
        step_args = dict(mode="immediate", frame=dict(duration=0, redraw=True),
                         transition=dict(duration=0))
        slider = dict(
            active=0, x=0.05, len=0.9, y=0.0, pad=dict(t=40, b=10),
            currentvalue=dict(prefix="Дата (UT): ", font=dict(size=16)),
            steps=[dict(method="animate", label=lab, args=[[lab], step_args])
                   for lab in labels],
        )
        if n_frames > 10:   # підписи поділок злилися б - лишаємо лише поточне значення
            slider["font"] = dict(color="rgba(0,0,0,0)")
        layout["sliders"] = [slider]
        layout["margin"] = dict(l=0, r=0, t=100, b=60)
        post_script = (PLAYER_JS
                       .replace("__NAMES__", json.dumps(labels, ensure_ascii=False))
                       .replace("__MS__", str(int(round(FRAME_SECONDS * 1000)))))
    fig.update_layout(**layout)

    # Збереження результату (для лінійної шкали додається суфікс, щоб не перезаписати "log")
    scale_tag = "" if COLOR_SCALE == "log" else "_linear"
    lon_tag = "_earth" if earth else ""
    d0, d1 = dates[0].strftime("%Y-%m-%d_%H%M"), dates[-1].strftime("%Y-%m-%d_%H%M")
    comet_tag = f"_{COMET_NAME.replace(' ', '')}" if COMET_NAME.strip() else ""
    common = (f"_{MODEL_TAG}_Br{lon_tag}{comet_tag}"
              f"_lon{LONGITUDE_CENTER_DEG:g}pm{LONGITUDE_HALF_WIDTH_DEG:g}deg"
              f"_theta{POLAR_ANGLE_MIN_DEG:g}-{POLAR_ANGLE_MAX_DEG:g}deg{scale_tag}")
    if n_frames == 1:
        output_html = DEST_FOLDER / f"br002_{d0}{common}_3D.html"
    else:
        output_html = DEST_FOLDER / f"br002_{d0}_to_{d1}_step{STEP_HOURS:g}h{common}_3D_anim.html"

    log.info("Зберігаю 3D-графік: %s", output_html)
    fig.write_html(str(output_html), auto_open=True, auto_play=False, post_script=post_script)
    log.info("Візуалізацію збережено (%.2f МБ).", mb(output_html.stat().st_size))
    return str(output_html)


# ==========================================
# ГОЛОВНА ФУНКЦІЯ
# ==========================================
def parse_datetime(value):
    """'YYYY-MM-DD HH:MM' або 'YYYY-MM-DD' (тоді 00:00); datetime повертається як є."""
    if isinstance(value, datetime):
        return value
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(value).strip(), fmt)
        except ValueError:
            pass
    raise ValueError(f"Не вдалося розпізнати дату/час {value!r}. "
                     "Очікується формат 'YYYY-MM-DD HH:MM' (напр. '2007-07-05 12:00').")


def make_date_list(start, end, step_hours):
    """Список дат від start до end включно з кроком step_hours."""
    if step_hours <= 0:
        raise ValueError("STEP_HOURS має бути більше нуля.")
    if end < start:
        raise ValueError(f"END ({end}) раніше за START ({start}).")
    step = timedelta(hours=step_hours)
    dates, t = [], start
    while t <= end:
        dates.append(t)
        t += step
    return dates


def duplicate_hint(first_label):
    if LONGITUDE_MODE == "fixed":
        return (f"Файл розв'язку один на оберт Керрінгтона, а сектор нерухомий у системі "
                f"Керрінгтона (LONGITUDE_MODE='fixed'), тому дати одного оберту (тут {first_label}) "
                "дають однакові кадри. Поставте LONGITUDE_MODE='earth' (сектор слідує за "
                "Сонцем) або збільште STEP_HOURS до ~655 год (один кадр на оберт).")
    return ("Однакові кадри в режимі 'earth' означають, що L0 і файл збіглися - "
            "перевірте STEP_HOURS і дати.")


def main():
    if COLOR_SCALE not in ("log", "linear"):
        raise ValueError("COLOR_SCALE має бути 'log' або 'linear'")
    if ON_DUPLICATE_DATA not in ("error", "skip", "warn"):
        raise ValueError("ON_DUPLICATE_DATA має бути 'error', 'skip' або 'warn'")
    if LONGITUDE_MODE not in ("earth", "fixed"):
        raise ValueError("LONGITUDE_MODE має бути 'earth' або 'fixed'")
    split_model()

    start, end = parse_datetime(START), parse_datetime(END)
    dates = make_date_list(start, end, STEP_HOURS)

    log.info("Лог-файл: %s", LOG_FILE)
    log.debug("NumPy %s | Plotly %s | Python %s",
              np.__version__, plotly.__version__, sys.version.split()[0])
    log.info("Модель: %s | розв'язок: %s", MODEL_ID, SOLUTION)
    log.info("Період: %s -> %s, крок %g год, дат: %d, зміна кадру кожні %g с, "
             "шкала: %s, довгота: %s",
             start, end, STEP_HOURS, len(dates), FRAME_SECONDS, COLOR_SCALE, LONGITUDE_MODE)
    if "hmi" in MODEL_ID.lower() and start < datetime(2010, 5, 1):
        log.warning("Модель «%s» - HMI-модель, а HMI працює лише з 2010 р.: для дат до цього "
                    "розв'язків немає (очікуйте HTTP 404). Для 2007 року використовуйте MDI-модель.",
                    MODEL_ID)
    t_start = time.perf_counter()

    cr_first, cr_last = carrington(start)[0], carrington(end)[0]
    log.info("Охоплені оберти Керрінгтона: CR %d ... CR %d (файлів: до %d)",
             cr_first, cr_last, cr_last - cr_first + 1)

    sectors, labels, ok_dates, skipped, summary = [], [], [], [], []
    seen_frames = {}
    loaded = (None, None)        # (шлях, дані HDF) - читаємо кожен файл оберту один раз
    for i, dt in enumerate(dates, 1):
        cr, l0 = carrington(dt)
        label = f"{dt:%Y-%m-%d %H:%M} (CR {cr}, L0={l0:.1f}°)"
        log.info("=" * 60)
        log.info("[%d/%d] %s UT -> CR %d, L0 = %.2f°", i, len(dates), dt.strftime("%Y-%m-%d %H:%M"),
                 cr, l0)
        try:
            path = ensure_hdf(cr)
            if loaded[0] != path:
                loaded = (path, load_br_hdf(path))
            phi, theta, radius, Br = loaded[1]
            sector = make_sector(phi, theta, radius, Br, l0, verbose=(not sectors))

            if sectors:
                ref = sectors[0]
                if (len(sector["rad"]) != len(ref["rad"])
                        or not np.allclose(sector["rad"], ref["rad"])
                        or not np.allclose(sector["pol"], ref["pol"])
                        or not np.allclose(sector["lon"], ref["lon"])):
                    raise ValueError("сітка цього файлу відрізняється від першої - "
                                     "кадри з різними сітками не можна об'єднати")

            # Перевірка: різні дати не повинні давати однакові кадри
            key = hashlib.sha256(np.asarray(sector["br"], dtype=np.float32).tobytes()).hexdigest()
            first = seen_frames.get(key)
            if first:
                msg = (f"Кадр для {label} ІДЕНТИЧНИЙ кадру для {first} (однакові значення Br, "
                       f"файл: {Path(path).name}). " + duplicate_hint(first))
                if ON_DUPLICATE_DATA == "error" and not COMET_NAME.strip():
                    raise DuplicateDataError(msg)
                if ON_DUPLICATE_DATA == "error" and COMET_NAME.strip():
                    log.warning("%s Br однаковий, але кадр лишається: позиція комети "
                                "запитується окремо для кожної дати.", msg)
                if ON_DUPLICATE_DATA == "skip" and not COMET_NAME.strip():
                    log.error("%s Дату пропущено.", msg)
                    skipped.append((label, f"ідентичний кадр {first}"))
                    continue
                if ON_DUPLICATE_DATA == "warn":
                    log.warning("%s Дату залишено (ON_DUPLICATE_DATA='warn').", msg)
        except DuplicateDataError:
            raise
        except DataMismatchError as e:
            log.error("Дату %s пропущено: %s", label, e)
            skipped.append((label, str(e)))
            continue
        except Exception as e:
            log.exception("Дату %s пропущено через помилку", label)
            skipped.append((label, f"помилка: {e}"))
            continue

        seen_frames.setdefault(key, label)
        sectors.append(sector)
        labels.append(label)
        ok_dates.append(dt)
        summary.append((label, cr, l0, Path(path).name, key[:8]))

    log.info("=" * 60)
    log.info("Підсумок (дата | оберт | L0 | файл | хеш кадру):")
    for lab, cr, l0, name, kh in summary:
        log.info("  %s | CR %d | L0=%.1f° | %s | %s", lab, cr, l0, name, kh)

    if not sectors:
        raise RuntimeError("Жодної дати не вдалося обробити - нічого візуалізувати.")
    if skipped:
        log.warning("Оброблено %d з %d дат. Пропущено:", len(sectors), len(dates))
        for lab, why in skipped:
            log.warning("  %s: %s", lab, why)

    comet_positions = []
    for dt, sector in zip(ok_dates, sectors):
        if not COMET_NAME.strip():
            comet_positions.append(None)
            continue
        try:
            pos = get_comet_position(dt, sector["l0"])
            comet_positions.append(pos)
            log.info("Комета %s | %s UT | φCarr=%.3f° | lat=%.3f° | θ=%.3f° | "
                     "r=%.3f R☉", COMET_NAME, dt, pos["carr_lon"], pos["lat"],
                     pos["theta"], pos["radius"])
        except Exception as e:
            log.exception("Не вдалося отримати положення комети %s на %s: %s",
                          COMET_NAME, dt, e)
            comet_positions.append(None)

    html_path = visualize_sectors(sectors, labels, ok_dates, comet_positions)

    log.info("Готово за %.2f с.", time.perf_counter() - t_start)
    log.info("HTML: %s", html_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("Скрипт завершився з помилкою")
        raise
