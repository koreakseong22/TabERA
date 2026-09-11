"""Render the supplied Sample #4 as a user-facing explanation snapshot.
Run: venv/Scripts/python.exe render_explanation_example.py
No distribution geometry or additional observations are synthesized.
"""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


def render(output):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42,
                         'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(17, 10.6))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0, 170), ylim=(106, 0))
    ax.axis('off')
    ink, muted = '#182B40', '#5C6C7C'
    bad, good, query = '#B34D20', '#2268AC', '#654BB0'
    pale, line = '#F5F7FA', '#DCE3EB'

    def text(x, y, s, size=11, color=ink, weight='normal', **kw):
        return ax.text(x, y, s, fontsize=size, color=color, weight=weight,
                       va='top', linespacing=1.4, **kw)

    def box(x, y, w, h, face='white', edge=line):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle='round,pad=0,rounding_size=1.2',
                     facecolor=face, edgecolor=edge, linewidth=.9))

    def rule(x, y, w):
        ax.plot([x, x+w], [y, y], color=line, lw=.8)

    def bar(x, y, w, fraction, color, height=1.1):
        ax.add_patch(Rectangle((x, y), w, height, color=line, lw=0))
        ax.add_patch(Rectangle((x, y), w*fraction, height, color=color, lw=0))

    text(5, 3, 'TabERA', 25, weight='bold')
    text(30, 4, 'Your prediction in context', 19)
    text(5, 10, 'SAMPLE #4', 10, muted, 'bold')
    box(5, 15, 160, 10, pale)
    text(8, 17, 'THIS CASE', 10, query, 'bold')
    text(28, 17, 'Duration  8     |     Credit amount  1,164     |     Purpose  Other     |     Checking status  <0', 12)
    text(28, 21.5, 'Selected details of the case being explained', 9, muted)

    box(5, 29, 42, 71)
    box(50, 29, 62, 71)
    box(115, 29, 50, 71)

    # Exact count display, not an embedding scatterplot.
    text(8, 32, 'A  Region', 17, weight='bold')
    text(8, 37, 'The group this case belongs to', 10, muted)
    text(8, 43, 'Region 3', 23, weight='bold')
    text(8, 49, '42 past cases', 12, muted)
    for i in range(42):
        ax.scatter(9.2 + (i % 7)*5.1, 56+(i//7)*3,
                   s=85, marker='o', color=bad if i < 35 else good,
                   edgecolors='none')
    text(8, 74, '35 bad · 83%', 12, bad, 'bold')
    text(28, 74, '7 good · 17%', 12, good, 'bold')
    text(8, 78, 'Each dot represents one past case.', 9, muted)
    rule(8, 82, 36)
    text(8, 85, 'A common purpose in this group', 10, muted)
    text(8, 89, 'New car', 15, weight='bold')
    text(43, 89, '36%', 15, weight='bold', ha='right')
    bar(8, 94, 36, .36, '#8595A8')
    text(8, 96, 'Most frequent purpose', 9, muted)

    text(53, 32, 'B  Evidence', 17, weight='bold')
    text(53, 37, 'Similar past cases in Region 3', 10, muted)
    for i in range(8):
        ax.scatter(54.3+i*3.6, 44, s=110,
                   color=bad if i < 6 else good, edgecolors='none')
    text(86, 42, '6 bad / 2 good', 11, weight='bold')
    text(53, 47, '75% bad · 25% good  |  8 cases found', 10, muted)

    def case(y, sid, label, score, heading, matches, differences, h):
        color = bad if label == 'bad' else good
        box(53, y, 56, h, pale)
        ax.add_patch(Rectangle((53, y+1.3), .4, h-2.6, color=color, lw=0))
        text(55, y+1, f'#{sid}  {label}', 13, color, 'bold')
        text(107, y+1.4, f'Similarity {score}', 9, muted, ha='right')
        text(55, y+4.4, heading, 9, color)
        text(55, y+7.3, matches, 9.1)
        text(55, y+7.3+len(matches.splitlines())*1.95, differences, 9.1)

    case(52, 278, 'bad', '0.715', 'Closest case',
         'Same: credit history = critical/other existing credit\n          employment = >=7',
         'Different checking status: <0 → 0<=X<200', 14.5)
    case(68, 43, 'bad', '0.639', 'Second closest case',
         'Same: other payment plans = bank\n          employment = >=7',
         'Different checking status: <0 → 0<=X<200', 14.5)
    case(84, 707, 'good', '0.637', 'Closest case with a different outcome',
         'Same: other payment plans = bank\n          job = high qualif/self emp/mgmt',
         'Different credit history: critical/other existing credit → all paid\nDifferent purpose: other → radio/tv', 15)

    text(118, 32, 'C  Prediction & Position', 17, weight='bold')
    text(118, 37, 'The result, and where this case stands', 10, muted)
    box(118, 42, 44, 13, '#FCF4EF', '#F0D9CA')
    text(120, 43.5, 'Predicted outcome', 10, muted)
    text(120, 47, 'bad', 25, bad, 'bold')
    text(160, 45.5, '89.1%', 29, bad, 'bold', ha='right')
    text(118, 57, 'Region baseline 89.0%  ·  Slightly stronger for bad', 9, muted)
    rule(118, 61, 44)
    text(118, 63, 'Compared with this region', 12, weight='bold')

    # These bars encode reported comparison shares, not raw-value density.
    def position(y, name, value, reference, above):
        text(118, y, f'{name}  {value}', 12, query, 'bold')
        text(162, y+.3, f'Reference {reference}', 9, muted, ha='right')
        text(118, y+3.5, f'{above:.0%} of region cases have a higher value', 9, muted)
        bar(118, y+6.6, 44, above, '#A99ACD', .9)
    position(68, 'Duration', '8', '30', 1.0)
    position(78, 'Credit amount', '1,164', '3,832', .95)
    text(118, 88, 'Purpose  Other', 12, query, 'bold')
    text(162, 88.3, '2% of this region', 10, muted, ha='right')
    bar(118, 93, 44, .02, query)
    text(118, 96, 'An uncommon purpose within this group', 9, muted)

    # A separate rank strip preserves the distance statistic without inventing
    # a region shape, density, or individual training-case positions.
    # Save a taller canvas to provide breathing room below the cards.
    ax.set_ylim(124, 0)
    fig.set_size_inches(17, 12.4)
    box(5, 103, 160, 16, pale)
    text(8, 105, 'Position relative to the region centre', 13, weight='bold')
    text(8, 111, 'Farther from the centre than 93%\nof the region’s past cases', 11, muted)
    bar(78, 111, 80, .93, '#C7BEDD', 1.3)
    marker_x = 78+80*.93
    ax.scatter(marker_x, 111.65, s=115, marker='D', color=query,
               edgecolors='white', linewidths=1.2, zorder=5)
    text(marker_x, 106.5, 'This case · 93%', 11, query, 'bold', ha='center')
    text(78, 114, 'Closer · 0%', 9, muted)
    text(158, 114, 'Farther · 100%', 9, muted, ha='right')
    text(5, 121, 'Case comparisons: this case → past case.   Position bars show regional comparison percentages.', 9, muted)

    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'pdf', 'svg'):
        path = output.with_suffix('.' + suffix)
        fig.savefig(path, dpi=300, facecolor='white')
        print(path)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path('docs/explanation_example/tabera_explanation_ui'))
    render(parser.parse_args().output)
