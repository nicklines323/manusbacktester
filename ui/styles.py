# src/ui/styles.py

"""
Styles for the dashboard.
"""

# Colors
COLORS = {
    'primary': '#375a7f',
    'secondary': '#444444',
    'success': '#00bc8c',
    'info': '#3498db',
    'warning': '#f39c12',
    'danger': '#e74c3c',
    'light': '#adb5bd',
    'dark': '#303030',
    'white': '#ffffff',
    'muted': '#6c757d'
}

# Styles
STYLES = {
    'card': {
        'background-color': COLORS['dark'],
        'border-color': COLORS['secondary'],
        'border-radius': '0.25rem',
        'box-shadow': '0 0.125rem 0.25rem rgba(0, 0, 0, 0.075)'
    },
    'card-header': {
        'background-color': COLORS['secondary'],
        'border-bottom': f'1px solid {COLORS["secondary"]}',
        'padding': '0.75rem 1.25rem'
    },
    'card-body': {
        'padding': '1.25rem'
    },
    'button': {
        'primary': {
            'background-color': COLORS['primary'],
            'border-color': COLORS['primary'],
            'color': COLORS['white']
        },
        'success': {
            'background-color': COLORS['success'],
            'border-color': COLORS['success'],
            'color': COLORS['white']
        },
        'danger': {
            'background-color': COLORS['danger'],
            'border-color': COLORS['danger'],
            'color': COLORS['white']
        }
    },
    'text': {
        'success': {
            'color': COLORS['success']
        },
        'danger': {
            'color': COLORS['danger']
        },
        'muted': {
            'color': COLORS['muted']
        }
    }
}
