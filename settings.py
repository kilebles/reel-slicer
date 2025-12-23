"""
Настройки для обработки видео
"""

# === Настройки наложения GIF ===

GIF_OVERLAY = {
    # Позиция наложения
    "x_position": "center",  # "center", "left", "right" или число в пикселях
    "y_position": -45,  # расстояние от верха в пикселях

    # Размер гифки
    "scale": 0.35,  # масштаб относительно ширины видео (0.3 = 30% ширины)

    # Анимация
    "frame_duration": 4.0,  # длительность показа каждого кадра (в секундах)
    "smooth_transitions": False,  # True = плавные переходы, False = резкая смена кадров
    "start_time": 1.0,  # когда начать показывать гифку (в секундах)
}

# === Настройки субтитров ===

SUBTITLES = {
    "model_size": "base",  # "tiny", "base", "small", "medium", "large"
    "words_per_line": 2,
    "font_size": 32,
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
