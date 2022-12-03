import enum


class ProductType(enum.Enum):
    THERMAL_MUG = "thermal_mug"
    DISHWASHER_TABLETS = "dishwasher_tablets"
    COLOR_WASHING_POWDER = "color_washing_powder"


class Sentiment(enum.Enum):
    POSITIVE = "POSITIVE"
    NEUTRAL = "NEUTRAL"
    NEGATIVE = "NEGATIVE"
