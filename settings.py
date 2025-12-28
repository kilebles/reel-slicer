"""
Настройки для обработки видео
"""

# === Настройки наложения GIF ===

GIF_OVERLAY = {
    # Позиция наложения
    "x_position": "center",  # "center", "left", "right" или число в пикселях
    "y_position": -55,  # расстояние от верха в пикселях (после letterbox)
    # Размер гифки
    "scale": 0.42,  # масштаб относительно ширины видео (0.3 = 30% ширины)
    # Letterbox (черные полосы сверху/снизу для освобождения места под гифку)
    "letterbox_enabled": True,  # включить черные полосы сверху/снизу
    "letterbox_top": 120,  # высота черной полосы сверху в пикселях
    "letterbox_bottom": 120,  # высота черной полосы снизу в пикселях
    # Анимация
    "frame_duration": 4.0,  # длительность показа каждого кадра (в секундах)
    "smooth_transitions": False,  # True = плавные переходы, False = резкая смена кадров
    "start_time": 1.0,  # когда начать показывать гифку (в секундах)
}

# === Настройки субтитров ===

SUBTITLES = {
    "model_size": "base",  # "tiny", "base", "small", "medium", "large"
    "words_per_line": 2,
    "font_size": 48,
    "font": "impact.ttf",
    "color": "white",
    "stroke_color": "black",
    "stroke_width": 3,
}

# === Настройки reframing (вертикальная обрезка) ===

REFRAMING = {
    "trigger_threshold": 40,
    "stop_threshold": 5,
    "ease_speed": 0.12,
}
