# Flöckchen – 3D-Modell (einfarbig)

Stilisiertes Modell der tonies-Figur „Flöckchen“ (Adventskalender 2025) für den 3D-Druck, z. B. auf einem Bambu-Lab-Drucker.

- `flockchen.stl` – druckfertig, wasserdicht, ein Körper, ca. 93 × 66 × 45 mm (H × B × T)
- `make_flockchen.py` – erzeugt das STL aus Signed-Distance-Grundformen (Kopf, Körper, Ohren, Füße, Arme, Gesicht, Lichterkette)
- `compare.png` – Vergleich mit den Referenzfotos

## Neu erzeugen

```sh
pip install numpy trimesh scikit-image
python3 make_flockchen.py
```

Die Größe stellt `mesh.apply_scale(0.75)` am Ende des Skripts ein.

## Drucktipps

- Auf den Füßen stehend drucken, Baumstützen für Ohren und Arme.
- Die Lichterkette ist ca. 1,6 mm dick und die empfindlichste Stelle.

Nur für den privaten Gebrauch; die Figur ist ein Produkt der tonies GmbH.
