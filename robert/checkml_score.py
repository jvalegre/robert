"""Shared CheckML score rounding and tier helpers."""

from decimal import Decimal, ROUND_HALF_UP

_TIER_COLORS = {
    'VERY WEAK': {'rgb': (1.0, 0.42, 0.42), 'text_hex': '#b00020'},
    'WEAK': {'rgb': (1.0, 0.55, 0.29), 'text_hex': '#c85f24'},
    'MODERATE': {'rgb': (1.0, 0.79, 0.38), 'text_hex': '#a77b16'},
    'STRONG': {'rgb': (0.60, 0.78, 0.95), 'text_hex': '#276dd6'},
    'VERY STRONG': {'rgb': (0.38, 0.60, 0.80), 'text_hex': '#1b5e20'},
    'UNKNOWN': {'rgb': (0.4, 0.4, 0.4), 'text_hex': '#666666'},
}


def round_checkml_score(score):
    """Round a 0-10 CheckML score to its displayed integer, using half-up rounding."""
    return int(Decimal(str(score)).quantize(Decimal('1'), rounding=ROUND_HALF_UP))


def checkml_score_tier(score):
    """Return the five-tier label used by the 0-10 CheckML score bars."""
    score = round_checkml_score(score)
    if score <= 2:
        return 'VERY WEAK'
    if score <= 4:
        return 'WEAK'
    if score <= 6:
        return 'MODERATE'
    if score <= 8:
        return 'STRONG'
    return 'VERY STRONG'


def checkml_tier_color(tier, output='rgb'):
    """Return a score tier/status fill RGB, CSS hex equivalent, or darker text hex."""
    if tier not in _TIER_COLORS:
        raise ValueError(f'unknown CheckML score tier: {tier!r}')
    palette = _TIER_COLORS[tier]
    if output == 'rgb':
        return palette['rgb']
    if output == 'text_hex':
        return palette['text_hex']
    if output == 'hex':
        return '#{:02x}{:02x}{:02x}'.format(*(round(channel * 255) for channel in palette['rgb']))
    raise ValueError("output must be 'rgb', 'hex', or 'text_hex'")


def checkml_score_color(score, output='rgb'):
    """Return the color for a score's rounded tier."""
    return checkml_tier_color(checkml_score_tier(score), output=output)
