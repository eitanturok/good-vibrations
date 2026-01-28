import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Color palette for sequential distinct colors
SEQUENTIAL_COLORS = list(mcolors.TABLEAU_COLORS.values())  # 10 distinct colors


def plot_gradients(data, n_rows=3, n_cols=3):
    """
    Interactive visualization showing multiple modes at once with shared position selector.

    Grid layout:
    - Top-left: Box position selector (10x7 grid + Empty row)
    - Other cells: Laser gradient quiver plots for modes 1 to (n_rows * n_cols - 1)

    Args:
        data: dict of experiment data with 'synced_mode_fft_gradients'
        n_rows: number of rows in the grid (default: 3)
        n_cols: number of columns in the grid (default: 3)

    Returns:
        fig: matplotlib figure with interactive controls
    """
    # Build position lookup: (x, y) -> data entry
    pos_to_data = {}
    empty_data = None

    for name, d in data.items():
        if d.get('duplicate_idx', 1) != 1:
            continue
        if 'synced_mode_fft_gradients' not in d:
            continue

        x, y = d.get('x_position'), d.get('y_position')
        if x is None or y is None:
            empty_data = d
        else:
            pos_to_data[(x, y)] = d

    # Get dimensions from first valid entry
    d0 = next(d for d in data.values() if 'synced_mode_fft_gradients' in d and d.get('duplicate_idx', 1) == 1)
    n_modes = min(n_rows * n_cols - 1, d0['synced_mode_fft_gradients'].shape[0])  # Show first (n_rows * n_cols - 1) modes
    n_lasers = d0['synced_mode_fft_gradients'].shape[2]  # 10
    mode_freqs = d0.get('mode_freqs', list(range(n_modes)))

    # Grid coordinates for laser plots
    X_laser, Y_laser = np.meshgrid(np.arange(n_lasers), np.arange(n_lasers))

    # Box grid dimensions
    n_box_x, n_box_y = 11, 8

    # State
    state = {
        'active_positions': set(),
        'empty_active': False,
        'color_idx': 0,
        'pos_colors': {},
        'pos_labels': [],
    }
    quivers = {}  # (mode_idx, pos) -> quiver object

    # Create figure with 3x3 grid
    fig = plt.figure(figsize=(10, 5))

    # Create GridSpec for 3x3 layout
    gs = fig.add_gridspec(n_rows, n_cols, top=0.92, bottom=0.05, left=0.05, right=0.98, hspace=0.25, wspace=0.15)

    # Suptitle with base text
    suptitle = fig.suptitle('Vibration Gradients  |  Positions: ', fontsize=14, fontweight='bold')

    # Top-left: Box position selector
    ax_grid = fig.add_subplot(gs[0, 0])
    ax_grid.set_xlim(-0.5, n_box_x - 0.5)
    ax_grid.set_ylim(-1.8, n_box_y - 0.5)
    ax_grid.set_aspect('equal')
    ax_grid.set_xlabel('Box X Position')
    ax_grid.set_ylabel('Box Y Position')
    ax_grid.set_title('Click to Select Positions', fontsize=10)
    ax_grid.set_xticks(range(n_box_x))
    ax_grid.set_yticks(range(-1, n_box_y))
    ax_grid.set_yticklabels(['Empty'] + [str(i) for i in range(n_box_y)])

    # Draw grid cells
    grid_rects = {}
    for x in range(n_box_x):
        for y in range(n_box_y):
            if (x, y) in pos_to_data:
                rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                     facecolor='#e0e0e0', edgecolor='#888888', linewidth=0.5)
                ax_grid.add_patch(rect)
                grid_rects[(x, y)] = rect
            else:
                rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                     facecolor='#f8f8f8', edgecolor='#cccccc', linewidth=0.5, hatch='//')
                ax_grid.add_patch(rect)

    # Empty row
    empty_rect = plt.Rectangle((-0.5, -1.5), n_box_x, 1,
                               facecolor='#d0d0d0', edgecolor='#888888', linewidth=1)
    ax_grid.add_patch(empty_rect)
    ax_grid.text(n_box_x / 2 - 0.5, -1, 'EMPTY', ha='center', va='center',
                fontsize=10, fontweight='bold', color='#555555')
    ax_grid.axhline(y=-0.5, color='#666666', linewidth=1.5, linestyle='-')

    # Create mode subplots (all cells except top-left which is the selector)
    mode_axes = []
    mode_titles = []

    # Generate grid positions for modes dynamically: skip (0,0) which is the selector
    mode_positions = [(r, c) for r in range(n_rows) for c in range(n_cols) if not (r == 0 and c == 0)]

    for mode_idx, (row, col) in enumerate(mode_positions[:n_modes]):
        ax = fig.add_subplot(gs[row, col])
        ax.set_xlim(-1.5, n_lasers + 0.5)
        ax.set_ylim(-1.5, n_lasers + 0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xticks(range(n_lasers))
        ax.set_yticks(range(n_lasers))

        freq = mode_freqs[mode_idx] if mode_idx < len(mode_freqs) else 0
        title = ax.set_title(f'Mode {mode_idx+1}: {freq:.1f} Hz', fontsize=10)

        mode_axes.append(ax)
        mode_titles.append(title)

    def get_next_color():
        color = SEQUENTIAL_COLORS[state['color_idx'] % len(SEQUENTIAL_COLORS)]
        state['color_idx'] += 1
        return color

    def compute_scale_for_mode(mode_idx):
        """Compute quiver scale for a specific mode."""
        max_mag = 0

        for pos in state['active_positions']:
            if pos in pos_to_data:
                d = pos_to_data[pos]
                dx = d['synced_mode_fft_gradients'][mode_idx, 0]
                dy = d['synced_mode_fft_gradients'][mode_idx, 1]
                max_mag = max(max_mag, np.max(np.sqrt(dx**2 + dy**2)))

        if state['empty_active'] and empty_data is not None:
            dx = empty_data['synced_mode_fft_gradients'][mode_idx, 0]
            dy = empty_data['synced_mode_fft_gradients'][mode_idx, 1]
            max_mag = max(max_mag, np.max(np.sqrt(dx**2 + dy**2)))

        return max_mag / 0.1 if max_mag > 0 else 1.0

    def update_suptitle():
        """Update figure suptitle with colored position labels."""
        # Clear old position labels
        for txt in state.get('pos_labels', []):
            txt.remove()
        state['pos_labels'] = []

        # Add colored position labels
        if state['active_positions'] or state['empty_active']:
            label_colors = []
            for pos in sorted(state['active_positions']):
                color = state['pos_colors'].get(pos, 'blue')
                label_colors.append((f'({pos[0]},{pos[1]})', color))
            if state['empty_active']:
                color = state['pos_colors'].get('empty', 'gray')
                label_colors.append(('Empty', color))

            x_start = 0.38
            for j, (label, color) in enumerate(label_colors):
                space = ' ' if j < len(label_colors) - 1 else ''
                txt = fig.text(x_start, 0.95, label + space, ha='left', va='bottom',
                              fontsize=11, fontweight='bold', color=color)
                state['pos_labels'].append(txt)
                x_start += len(label + space) * 0.005
        else:
            txt = fig.text(0.38, 0.95, 'None', ha='left', va='bottom',
                          fontsize=11, fontstyle='italic', color='gray')
            state['pos_labels'].append(txt)

    def draw_quivers():
        """Redraw all quivers for all modes."""
        # Clear existing quivers
        for q in quivers.values():
            q.remove()
        quivers.clear()

        # Update suptitle
        update_suptitle()

        if not state['active_positions'] and not state['empty_active']:
            fig.canvas.draw_idle()
            return

        # Draw quivers for each mode
        for mode_idx, ax in enumerate(mode_axes):
            scale = compute_scale_for_mode(mode_idx)

            for pos in state['active_positions']:
                if pos not in pos_to_data:
                    continue
                d = pos_to_data[pos]
                color = state['pos_colors'].get(pos, 'blue')

                dx = d['synced_mode_fft_gradients'][mode_idx, 0]
                dy = d['synced_mode_fft_gradients'][mode_idx, 1]

                q = ax.quiver(X_laser, Y_laser, dx, dy, scale=scale, color=color, alpha=0.7)
                quivers[(mode_idx, pos)] = q

            if state['empty_active'] and empty_data is not None:
                color = state['pos_colors'].get('empty', 'gray')
                dx = empty_data['synced_mode_fft_gradients'][mode_idx, 0]
                dy = empty_data['synced_mode_fft_gradients'][mode_idx, 1]

                q = ax.quiver(X_laser, Y_laser, dx, dy, scale=scale, color=color, alpha=0.7)
                quivers[(mode_idx, 'empty')] = q

        fig.canvas.draw_idle()

    def toggle_position(pos):
        """Toggle a box position on/off."""
        if pos in state['active_positions']:
            state['active_positions'].remove(pos)
            if pos in state['pos_colors']:
                del state['pos_colors'][pos]
            if pos in grid_rects:
                grid_rects[pos].set_facecolor('#e0e0e0')
        else:
            if pos in pos_to_data:
                state['active_positions'].add(pos)
                color = get_next_color()
                state['pos_colors'][pos] = color
                if pos in grid_rects:
                    grid_rects[pos].set_facecolor(color)

        draw_quivers()

    def toggle_empty():
        """Toggle empty position on/off."""
        if state['empty_active']:
            state['empty_active'] = False
            if 'empty' in state['pos_colors']:
                del state['pos_colors']['empty']
            empty_rect.set_facecolor('#d0d0d0')
        else:
            state['empty_active'] = True
            color = get_next_color()
            state['pos_colors']['empty'] = color
            empty_rect.set_facecolor(color)

        draw_quivers()

    def on_click(event):
        """Handle click on grid."""
        if event.inaxes != ax_grid:
            return
        if event.xdata is None or event.ydata is None:
            return

        x_idx = int(round(event.xdata))
        y_idx = int(round(event.ydata))

        # Check if clicking on empty row
        if -1.5 <= event.ydata <= -0.5:
            toggle_empty()
            return

        # Check bounds for regular grid
        if 0 <= x_idx < n_box_x and 0 <= y_idx < n_box_y:
            pos = (x_idx, y_idx)
            if pos in pos_to_data:
                toggle_position(pos)

    # Connect event handler
    fig.canvas.mpl_connect('button_press_event', on_click)

    # Store references
    fig._widgets = {
        'grid_rects': grid_rects,
        'empty_rect': empty_rect,
    }
    fig._state = state
    fig._quivers = quivers

    # Initial suptitle update
    update_suptitle()

    return fig
