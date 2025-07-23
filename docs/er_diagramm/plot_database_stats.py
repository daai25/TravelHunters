import sqlite3
import os
import matplotlib.pyplot as plt

# Datenbanken und Pfad
db_dir = '../../data_acquisition/database'
db_files = [
    'byg_images.db',
    'city_images.db',
    'geo_city_images.db',
    'wiki_images.db'
]

def count_city_images(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT COUNT(*) FROM city_images")
        count = cursor.fetchone()[0]
    except Exception:
        count = 0
    conn.close()
    return count

# Zähle Bilder pro DB
counts = []
for db in db_files:
    db_path = os.path.join(db_dir, db)
    counts.append((db, count_city_images(db_path)))

# Sortiere absteigend
counts.sort(key=lambda x: x[1], reverse=True)
db_names = [x[0] for x in counts]
db_counts = [x[1] for x in counts]

# Farben
colors = plt.cm.tab10.colors
bar_colors = [colors[i % len(colors)] for i in range(len(db_names))]

# Gebrochene y-Achse
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7,5), gridspec_kw={'height_ratios': [2, 1]})

# Oberer Plot (ab 1000)
ax1.bar(db_names, db_counts, color=bar_colors)
ax1.set_ylim(1000, max(db_counts) * 1.1)
ax1.spines['bottom'].set_visible(False)
ax1.tick_params(labeltop=False)

# Unterer Plot (bis 1000)
ax2.bar(db_names, db_counts, color=bar_colors)
ax2.set_ylim(0, 150)
ax2.spines['top'].set_visible(False)

# "Zacken" für Achsenbruch
d = .015
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((-d, +d), (-d, +d), **kwargs)
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
kwargs.update(transform=ax2.transAxes)
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

fig.suptitle('Images per database (table: city_images)')
ax2.set_ylabel('Number of images')
plt.xticks(rotation=15)
plt.tight_layout()
plt.subplots_adjust(hspace=0.05)
# plt.show()
plt.savefig('city_images.png')
plt.close()