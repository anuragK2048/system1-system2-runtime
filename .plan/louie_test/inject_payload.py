import xml.etree.ElementTree as ET
from pathlib import Path

SRC = Path("mujoco_menagerie/skydio_x2/x2.xml")   # or scene.xml's drone file
OUT = Path("models/skydio_x2_payload.xml")

tree = ET.parse(SRC)
# find the root body = the one containing the freejoint
root_body = next(b for b in tree.getroot().iter("body")
                 if b.find("freejoint") is not None or b.find("joint[@type='free']") is not None)

payload = ET.SubElement(root_body, "body", {"name": "payload", "pos": "0 0 -0.06"})
ET.SubElement(payload, "joint", {"name": "payload_roll",  "type": "hinge",
                                 "axis": "1 0 0", "damping": "0.005"})
ET.SubElement(payload, "joint", {"name": "payload_pitch", "type": "hinge",
                                 "axis": "0 1 0", "damping": "0.005"})
ET.SubElement(payload, "inertial", {"pos": "0 0 -0.35", "mass": "0.30",
                                    "diaginertia": "0.001 0.001 0.001"})
ET.SubElement(payload, "geom", {"name": "cable", "type": "cylinder",
                                "fromto": "0 0 0  0 0 -0.35", "size": "0.005",
                                "rgba": "0.8 0.8 0.8 1", "density": "0"})
ET.SubElement(payload, "geom", {"name": "payload_box", "type": "box",
                                "pos": "0 0 -0.35", "size": "0.05 0.05 0.03",
                                "rgba": "1 0.5 0 1", "density": "0"})
tree.write(OUT)
print(f"[OK] wrote {OUT} — cable 0.35 m, mass 0.30 kg, "
      f"pendulum freq = {(9.81/0.35)**0.5/(2*3.14159):.2f} Hz")