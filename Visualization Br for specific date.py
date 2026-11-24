import sys
import time
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import plotly
import plotly.graph_objects as go
from pyhdf.SD import SD, SDC
from playwright.sync_api import sync_playwright

# ==========================================
# НАЛАШТУВАННЯ
# ==========================================
BASE_URL = "https://www.predsci.com/mhdweb/data_access.php"

# Папка, куди зберігати файли
DEST_FOLDER = Path(r"C:\Users\gusar\OneDrive\Desktop\Vitishka data from MHD")

# Політропна корональна модель + політропна геліосферна модель
# (значення #ps-data-id; None - залишити те, що стоїть на сторінці за замовчуванням)
MODEL_ID = "high/hmi_mast_mas_std_0201"
MODEL_NAME = "hmi__high-cor-thermo2-exp01__high-hel-poly-exp01"
MODEL_TAG = "thermo2_hel-poly"

# Опис рядка, з якого беремо посилання
ROW_TEXT = "Radial component of magnetic field"

# --- Дата і час моделювання (UT) ---
YEAR = 2024
MONTH = 6
DAY = 20
HOUR = 12
MINUTE = 0

# Розв'язок: "helio" - геліосферний, "corona" - корональний
SOLUTION = "helio"

# Якщо файл за цією датою вже завантажено - не завантажувати повторно
REUSE_EXISTING = True

# --- Параметри візуалізації ---
# Радіальний діапазон, R_sun
R_MIN = 30.0
R_MAX = 40.0

# Довгота: центр і півширина сектора, градуси
LONGITUDE_CENTER_DEG = 0.0
LONGITUDE_HALF_WIDTH_DEG = 50.0

# Полярний кут (колатитуда, 0° = північний полюс, 90° = екватор), градуси
POLAR_ANGLE_MIN_DEG = 0.0
POLAR_ANGLE_MAX_DEG = 180.0

# Перехід від кодових одиниць MAS (так записано Br у HDF) до нанотесла.
# B0 = V0*sqrt(4*pi*rho0) = 2.20689 Гс (нормування MAS, psi_io.units.FN_B);
# 1 Гс = 1e5 нТ  ->  1 кодова одиниця = 2.20689e5 нТ
BR_CODE_TO_NT = 2.2068914e5

# Логарифмічна кольорова шкала для Br (нТ).
# Br має знак, тому використовується СИМЕТРИЧНИЙ логарифм (symlog):
#   колір ~ sign(Br) * log10(1 + |Br| / SYMLOG_LINTHRESH_NT)
# Синій - від'ємні, червоний - додатні; на шкалі підписані справжні значення в нТ.
# |Br| нижче порога відображаються лінійно біля нуля (щоб не було log(0)).
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


def make_filename(year, month, day, hour, minute):
    return f"br002_{year}-{month}-{day}_{hour}{minute}UT.hdf"


# ==========================================
# ЗАВАНТАЖЕННЯ З MHDweb
# ==========================================
def find_download_link(page, label_text=ROW_TEXT):
    """
    Шукає на сторінці текст-опис і повертає locator найближчого
    за ним посилання з іконкою завантаження.
    Не залежить від того, чи є на сторінці справжні <tr>.
    """
    label = page.get_by_text(label_text, exact=True)
    n = label.count()
    log.info("Елементів із точним текстом «%s»: %d", label_text, n)

    if n == 0:
        label = page.locator(f"text={label_text}")
        n = label.count()
        log.info("Елементів за частковим збігом тексту: %d", n)

    if n == 0:
        dump = LOG_DIR / "page_dump.html"
        dump.write_text(page.content(), encoding="utf-8")
        log.error("Текст «%s» не знайдено. HTML сторінки збережено: %s", label_text, dump)
        log.error("Кількість frame на сторінці: %d", len(page.frames))
        for fr in page.frames:
            log.error("  frame: %s", fr.url)
        raise RuntimeError(f"Не знайдено елемент із текстом «{label_text}»")

    tag = label.first.evaluate("e => e.tagName")
    log.debug("Перший збіг: <%s>", tag)

    # Перше посилання з іконкою завантаження після цього тексту
    link = label.first.locator(
        "xpath=following::a[.//img[contains(@src,'download')]][1]"
    )
    if link.count() == 0:
        log.warning("Посилання з іконкою download не знайдено, беру перше <a> після тексту")
        link = label.first.locator("xpath=following::a[1]")

    href = link.get_attribute("href")
    log.info("Посилання: %s", href)
    if not href:
        raise RuntimeError("У знайденого посилання немає href")
    if not href.rstrip().endswith("br002.hdf"):
        log.warning("Посилання не закінчується на br002.hdf - перевір, що це саме Br")
    return link


def download_br_from_mhdweb(month, day, year, hour, minute,
                            solution="helio", headless=True,
                            pause_for_check=False):
    """
    Відкриває MHDweb Data Access, вводить дату/час, вмикає radio-перемикач
    розв'язку (id="helio" або id="corona"), перевіряє, що посилання
    на "Radial component of magnetic field" відповідає вибраному розв'язку,
    і завантажує файл через браузер (expect_download).

    Повертає шлях до збереженого файлу.
    """
    if solution not in ("helio", "corona"):
        raise ValueError("solution має бути 'helio' або 'corona'")

    log.info("Дата/час: %s-%s-%s %s:%s UT | розв'язок: %s | headless=%s",
             year, month, day, hour, minute, solution, headless)

    dest_path = DEST_FOLDER / make_filename(year, month, day, hour, minute)

    with sync_playwright() as p:
        t0 = time.perf_counter()
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page(accept_downloads=True)
        log.debug("Браузер запущено за %.2f с", time.perf_counter() - t0)

        # Події браузера - у файл логу
        page.on("console", lambda m: log.debug("[browser console] %s: %s", m.type, m.text))
        page.on("pageerror", lambda e: log.warning("[browser pageerror] %s", e))
        page.on("requestfailed",
                lambda r: log.warning("[browser requestfailed] %s (%s)", r.url, r.failure))

        try:
            # --- Відкриття сторінки ---
            t0 = time.perf_counter()
            resp = page.goto(BASE_URL, wait_until="networkidle")
            log.info("Сторінку відкрито за %.2f с, HTTP %s",
                     time.perf_counter() - t0, resp.status if resp else "?")

            # --- Вибір моделі ---
            if MODEL_ID:
                page.locator("#ps-data-id").wait_for()
                page.locator("#ps-data-id").select_option(value=MODEL_ID)
                page.wait_for_timeout(1000)
                log.info("Вибрана модель: %s", MODEL_ID)

            # --- Дата і час ---
            for field_id, value in [
                ("ps-month", month),
                ("ps-day", day),
                ("ps-year", year),
                ("ps-hour", hour),
                ("ps-minute", minute),
            ]:
                field = page.locator(f"#{field_id}")
                field.fill(value)
                # сайт слухає "change" (onchange="psSetDate(...)")
                field.dispatch_event("change")
                page.wait_for_timeout(500)  # час на AJAX
                log.debug("#%s = %s (на сторінці: %s)",
                          field_id, value, field.input_value())

            # --- Вибір розв'язку: <input type="radio" id="helio" name="solution" ...> ---
            radio = page.locator(f"#{solution}")
            radio.wait_for()
            log.info("Radio #%s до кліку: checked=%s", solution, radio.is_checked())

            if not radio.is_checked():
                # check() робить реальний клік, тож спрацьовує onchange="setSolution();"
                radio.check()
                page.wait_for_timeout(500)
                log.info("Radio #%s ввімкнено кліком", solution)
            else:
                # Уже стоїть "checked", але setSolution() міг не виконатися -
                # викликаємо подію явно, щоб сайт оновив посилання
                radio.dispatch_event("change")
                page.wait_for_timeout(500)
                log.info("Radio #%s уже вибрано, надіслано подію change", solution)

            if not radio.is_checked():
                raise RuntimeError(f"Не вдалося вибрати radio #{solution}")
            log.info("OK: radio #%s вибрано (value=%s)", solution, radio.get_attribute("value"))

            # --- Якщо є кнопка "Update Data" - тиснемо ---
            for selector in ("input[value='Update Data']", "text=Update Data"):
                try:
                    page.click(selector, timeout=3000)
                    log.info("Натиснуто «Update Data» (%s)", selector)
                    break
                except Exception:
                    log.debug("Кнопку «Update Data» не знайдено за селектором %s", selector)

            page.wait_for_load_state("networkidle")

            # --- Посилання на Br ---
            link = find_download_link(page, ROW_TEXT)
            href = link.get_attribute("href")

            # Перевірка, що посилання справді для потрібного розв'язку
            expected_part = f"/{solution}/"
            if href and expected_part not in href:
                log.warning(
                    "У посиланні немає «%s» - сайт, можливо, ще не оновив дані "
                    "після зміни розв'язку/дати", expected_part)
                page.wait_for_timeout(1500)
                href = link.get_attribute("href")
                log.info("Посилання після додаткового очікування: %s", href)
                if href and expected_part not in href:
                    raise RuntimeError(
                        f"Посилання {href} не відповідає розв'язку '{solution}'")

            if pause_for_check:
                log.warning("Перевір дату/розв'язок на сторінці й натисни Resume в Inspector")
                page.pause()

            # --- Завантаження через браузер ---
            t0 = time.perf_counter()
            with page.expect_download() as download_info:
                link.click()
                log.info("Клік по посиланню, чекаю на завантаження...")
            download = download_info.value
            log.info("Сервер віддав файл: %s", download.suggested_filename)
            log.debug("URL завантаження: %s", download.url)

            download.save_as(str(dest_path))
            dt = time.perf_counter() - t0
            size = dest_path.stat().st_size
            log.info("Збережено: %s (%.2f МБ за %.2f с, ~%.2f МБ/с)",
                     dest_path, mb(size), dt, mb(size) / max(dt, 1e-9))
            if size == 0:
                raise RuntimeError("Завантажений файл порожній")

            return str(dest_path)

        except Exception:
            log.exception("Помилка під час завантаження")
            try:
                shot = LOG_DIR / f"error_{time.strftime('%H-%M-%S')}.png"
                page.screenshot(path=str(shot), full_page=True)
                log.error("Знімок сторінки збережено: %s", shot)
            except Exception:
                pass
            raise
        finally:
            browser.close()
            log.debug("Браузер закрито")


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
# ВІЗУАЛІЗАЦІЯ
# ==========================================
def nice_degree_ticks(lo, hi, wrap=False, max_ticks=8):
    """
    Підбирає «круглі» поділки осі в градусах.
    wrap=True - підписи приводяться до 0-360° (вісь при цьому лишається неперервною).
    """
    span = max(hi - lo, 1e-9)
    step = next((s for s in (1, 2, 5, 10, 15, 20, 30, 45, 60, 90)
                 if span / s <= max_ticks), 90)
    vals = np.arange(np.ceil(lo / step) * step, hi + 1e-9, step)
    texts = [f"{(v % 360 if wrap else v):g}°" for v in vals]
    return vals.tolist(), texts


def spherical_graticule(lon_vals, lon_texts, pol_vals, pol_texts,
                        lon_lo, lon_hi, pol_lo, pol_hi, r_shell):
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
        lt.append(f"φ={t}")

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


def prepare_log_colors(br):
    """
    Повертає (значення_кольору, cmax, налаштування_colorbar) для логарифмічної шкали.
    Підписи на colorbar - справжні значення Br у нТ.
    """
    abs_br = np.abs(br)
    nz = abs_br[abs_br > 0]
    if SYMLOG_LINTHRESH_NT:
        lin = float(SYMLOG_LINTHRESH_NT)
    elif nz.size:
        lin = 10.0 ** np.floor(np.log10(max(float(np.percentile(nz, 5)), 1e-9)))
    else:
        lin = 1.0

    br_max = float(np.nanmax(abs_br))
    vals = symlog(br, lin)
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
    return np.asarray(vals, dtype=float), cmax, cfg


def visualize_br_sector(file_path, date_label):
    """
    3D-візуалізація вихідних вузлів MHD-сітки.
    Колір точок показує Br.
    Довгота та полярний кут задаються в градусах.
    """
    # 1. Зчитування HDF
    phi, theta, radius, Br = load_br_hdf(file_path)

    # 2. Сітка сферичних координат (порядок: phi, theta, radius)
    PHI, THETA, R = np.meshgrid(phi, theta, radius, indexing="ij")

    # 3. Кути у градусах (осі графіка: довгота, полярний кут, радіус)
    LON_GRID = np.degrees(PHI)
    POLAR_GRID = np.degrees(THETA)

    # Коректна різниця довгот з урахуванням 0°/360°
    lon_diff = ((LON_GRID - LONGITUDE_CENTER_DEG + 180) % 360) - 180

    # Довгота для осі: неперервна навколо центру сектора
    # (напр. центр 0°, ширина ±90° -> вісь від -90° до +90°, без стрибка 360°->0°)
    LON_PLOT = LONGITUDE_CENTER_DEG + lon_diff

    # 5. Відбір вузлів за радіусом і кутами (з підрахунком по умовах)
    cond_r = (R >= R_MIN) & (R <= R_MAX)
    cond_lon = np.abs(lon_diff) <= LONGITUDE_HALF_WIDTH_DEG
    cond_pol = (POLAR_GRID >= POLAR_ANGLE_MIN_DEG) & (POLAR_GRID <= POLAR_ANGLE_MAX_DEG)
    cond_fin = np.isfinite(Br)

    log.info("Доступний діапазон: R=[%.2f, %.2f] R_sun, довгота=[%.1f, %.1f]°, "
             "полярний кут=[%.1f, %.1f]°",
             radius.min(), radius.max(),
             LON_GRID.min(), LON_GRID.max(),
             POLAR_GRID.min(), POLAR_GRID.max())
    for nm, c in [("радіус", cond_r), ("довгота", cond_lon),
                  ("полярний кут", cond_pol), ("скінченний Br", cond_fin)]:
        log.info("  умова «%s»: %d вузлів (%.2f%%)", nm, int(c.sum()), 100.0 * c.mean())

    mask = cond_r & cond_lon & cond_pol & cond_fin

    lon = LON_PLOT[mask]        # довгота для осі (неперервна)
    lon_true = LON_GRID[mask]   # справжня довгота 0-360° (для підказки)
    pol = POLAR_GRID[mask]      # полярний кут
    rad = R[mask]               # радіус
    br = Br[mask] * BR_CODE_TO_NT   # нТ

    log.info("Кількість вузлів для візуалізації: %d", len(br))

    if len(br) == 0:
        raise ValueError(
            "У вибраній області немає вузлів. "
            f"Доступно: R=[{radius.min():.1f}, {radius.max():.1f}] R_sun, "
            f"полярний кут=[{POLAR_GRID.min():.1f}, {POLAR_GRID.max():.1f}]°. "
            "Перевір межі радіуса, довготи та полярного кута."
        )

    log_array("Br (вибране, нТ)", br)

    # 6. Логарифмічна (симетрична) кольорова шкала
    color_vals, cmax, colorbar_cfg = prepare_log_colors(br)

    fig = go.Figure()

    # 7. Координати точок залежно від режиму
    if PLOT_MODE == "spherical":
        theta_r = np.radians(pol)
        lon_r = np.radians(lon_true)
        px = rad * np.sin(theta_r) * np.cos(lon_r)
        py = rad * np.sin(theta_r) * np.sin(lon_r)
        pz = rad * np.cos(theta_r)
    elif PLOT_MODE == "angles":
        px, py, pz = lon, pol, rad
    else:
        raise ValueError("PLOT_MODE має бути 'spherical' або 'angles'")
    log.info("Режим графіка: %s", PLOT_MODE)

    fig.add_trace(go.Scatter3d(
        x=px,
        y=py,
        z=pz,
        mode="markers",
        marker=dict(
            size=MARKER_SIZE,
            color=color_vals,   # колір за symlog(Br); підказка показує справжній Br
            colorscale="RdBu",
            reversescale=True,   # синій = від'ємні, червоний = додатні
            cmin=-cmax,
            cmax=cmax,
            colorbar=colorbar_cfg,
            opacity=MARKER_OPACITY
        ),
        text=[
            f"φ (довгота) = {ll % 360:.3f}°<br>"
            f"θ (полярний кут) = {tt:.3f}°<br>"
            f"r = {rr:.3f} R☉<br>"
            f"Br = {bb:.6g} нТ"
            for ll, tt, rr, bb in zip(lon_true, pol, rad, br)
        ],
        hovertemplate="%{text}<extra></extra>",
        name="Вузли MHDweb"
    ))

    # 8. Діапазони та поділки у градусах
    lon_lo, lon_hi = float(lon.min()), float(lon.max())
    pol_lo, pol_hi = float(pol.min()), float(pol.max())
    log.info("Діапазони: довгота [%.2f, %.2f]°, полярний кут [%.2f, %.2f]°, "
             "r [%.2f, %.2f] R☉",
             lon_lo, lon_hi, pol_lo, pol_hi, float(rad.min()), float(rad.max()))

    lon_tickvals, lon_ticktext = nice_degree_ticks(lon_lo, lon_hi, wrap=True)
    pol_tickvals, pol_ticktext = nice_degree_ticks(pol_lo, pol_hi)

    if PLOT_MODE == "spherical":
        # Сферична сітка на зовнішній оболонці вибраного шару
        r_shell = float(rad.max())
        grat_lines, grat_labels = spherical_graticule(
            lon_tickvals, lon_ticktext, pol_tickvals, pol_ticktext,
            lon_lo, lon_hi, pol_lo, pol_hi, r_shell,
        )
        fig.add_trace(grat_lines)
        fig.add_trace(grat_labels)
        log.info("Сітка: %d меридіанів, %d паралелей на r=%.2f R☉",
                 len(lon_tickvals), len(pol_tickvals), r_shell)

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
            xaxis=dict(title="Довгота φ (°)",
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

    # 9. Заголовок і підписи осей
    fig.update_layout(
        title=(
            f"Радіальна компонента Br | {MODEL_NAME}<br>"
            f"{date_label} UT | "
            f"R = {R_MIN}–{R_MAX} R☉ | "
            f"φ = {LONGITUDE_CENTER_DEG}° ± "
            f"{LONGITUDE_HALF_WIDTH_DEG}° | "
            f"θ = {POLAR_ANGLE_MIN_DEG}°–"
            f"{POLAR_ANGLE_MAX_DEG}°"
        ),
        scene=scene_cfg,
        margin=dict(l=0, r=0, t=100, b=0)
    )

    # 10. Збереження результату
    stem = Path(file_path).stem

    output_html = DEST_FOLDER / (
        f"{stem}_{MODEL_TAG}_Br"
        f"_lon{LONGITUDE_CENTER_DEG:g}"
        f"pm{LONGITUDE_HALF_WIDTH_DEG:g}deg"
        f"_theta{POLAR_ANGLE_MIN_DEG:g}"
        f"-{POLAR_ANGLE_MAX_DEG:g}deg_3D.html"
    )

    log.info("Зберігаю 3D-графік: %s", output_html)
    fig.write_html(str(output_html), auto_open=True)
    log.info("Візуалізацію збережено (%.2f МБ).", mb(output_html.stat().st_size))
    return str(output_html)


# ==========================================
# ГОЛОВНА ФУНКЦІЯ
# ==========================================
def main():
    # Дата і час беруться з налаштувань (YEAR, MONTH, DAY, HOUR, MINUTE) - без введення в терміналі
    try:
        datetime(int(YEAR), int(MONTH), int(DAY), int(HOUR), int(MINUTE))
    except ValueError as e:
        raise ValueError(
            f"Некоректна дата/час у налаштуваннях: {YEAR}-{MONTH}-{DAY} {HOUR}:{MINUTE} ({e})"
        ) from e

    month = f"{int(MONTH):02d}"
    day = f"{int(DAY):02d}"
    year = f"{int(YEAR):04d}"
    hour = f"{int(HOUR):02d}"
    minute = f"{int(MINUTE):02d}"

    solution = SOLUTION
    date_label = f"{year}-{month}-{day} {hour}:{minute}"

    log.info("Лог-файл: %s", LOG_FILE)
    log.debug("NumPy %s | Plotly %s | Python %s",
              np.__version__, plotly.__version__, sys.version.split()[0])
    t_start = time.perf_counter()

    # --- 1. Завантаження (або використання вже наявного файлу) ---
    existing = DEST_FOLDER / make_filename(year, month, day, hour, minute)
    if REUSE_EXISTING and existing.exists() and existing.stat().st_size > 0:
        log.info("Файл уже є, завантаження пропущено: %s", existing)
        path = str(existing)
    else:
        path = download_br_from_mhdweb(
            month, day, year, hour, minute,
            solution=solution,
            headless=True,          # False - щоб бачити браузер
            pause_for_check=False,  # True - зупинка перед завантаженням
        )

    # --- 2. Візуалізація ---
    html_path = visualize_br_sector(path, date_label)

    log.info("Готово за %.2f с.", time.perf_counter() - t_start)
    log.info("HDF:  %s", path)
    log.info("HTML: %s", html_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("Скрипт завершився з помилкою")
        raise