"""Re-fetch pinned Stats NZ Accommodation Supplement and joint-geography pages.

``tools/build_nz_as_area_crosswalk.py`` reads its inputs from a local directory
and verifies every byte against the pins in ``nz/as_area_crosswalk.json``. The
pages are public Stats NZ map-service responses (CC BY 4.0), too large to
commit, so this tool recreates that directory: it fetches each pinned page from
its recorded query URL, and each layer's metadata from the layer endpoint, and
refuses any response whose size or SHA-256 differs from its pin. A mismatch
means the publisher's service now returns different bytes; regenerate the
crosswalk deliberately rather than relaxing the pin. Example::

    python tools/fetch_nz_as_area_sources.py --out nz-as-area-sources
    python tools/build_nz_as_area_crosswalk.py --sources nz-as-area-sources \
        --out packages/microcosm-build/src/microcosm/build/nz/as_area_crosswalk.json

Use ``--joint-geography`` for the 29 Meshblock Higher Geographies 2026 pages
carrying the joint SA1, TA and REGC concordance. Their receipt pins below are
also consumed by the joint-support generator. ``--out`` defaults to
``NZ_JOINT_GEOGRAPHY_SOURCES`` in that mode, or ``NZ_AS_AREA_SOURCES`` otherwise.

Only public URLs are requested and no credentials are sent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

SPEC = Path("packages/microcosm-build/src/microcosm/build/nz/as_area_crosswalk.json")
USER_AGENT = "PolicyEngine-microcosm/fetch_nz_as_area_sources"


# Stats NZ MBHG 2026, CC BY 4.0; receipts collected 2026-10-24.
JOINT_GEOGRAPHY_PROVENANCE = {
    "pages": [
        {
            "file": "mbhg-2026-regc-p000.json",
            "url": "https://services2.arcgis.com/vKb0s8tBIA3bdocZ/arcgis/rest/services/Meshblock_Higher_Geographies_2026/FeatureServer/0/query?where=1%3D1&outFields=MB2026_V1_00,SA12026_V1_00,TA2026_V1_00,TA2026_V1_00_NAME,REGC2025_V1_00,REGC2025_V1_00_NAME,TALB2026_V1_00,TALB2026_V1_00_NAME&returnGeometry=false&f=json&orderByFields=OBJECTID&resultOffset=0&resultRecordCount=2000",
            "sha256": "e1e984283d49f4dba3757144e40b699796bf1c2ea1a34fb524015b56ba8a0b70",
            "bytes": 521003,
            "features": 2000,
        },
        {
            "file": "mbhg-2026-regc-p001.json",
            "url": "https://services2.arcgis.com/vKb0s8tBIA3bdocZ/arcgis/rest/services/Meshblock_Higher_Geographies_2026/FeatureServer/0/query?where=1%3D1&outFields=MB2026_V1_00,SA12026_V1_00,TA2026_V1_00,TA2026_V1_00_NAME,REGC2025_V1_00,REGC2025_V1_00_NAME,TALB2026_V1_00,TALB2026_V1_00_NAME&returnGeometry=false&f=json&orderByFields=OBJECTID&resultOffset=2000&resultRecordCount=2000",
            "sha256": "33ad698c6e732312900b6303fa57e82ec5ea0f220038d583ff40d694a5a42241",
            "bytes": 514300,
            "features": 2000,
        },
        {
            "file": "mbhg-2026-regc-p002.json",
            "url": "https://services2.arcgis.com/vKb0s8tBIA3bdocZ/arcgis/rest/services/Meshblock_Higher_Geographies_2026/FeatureServer/0/query?where=1%3D1&outFields=MB2026_V1_00,SA12026_V1_00,TA2026_V1_00,TA2026_V1_00_NAME,REGC2025_V1_00,REGC2025_V1_00_NAME,TALB2026_V1_00,TALB2026_V1_00_NAME&returnGeometry=false&f=json&orderByFields=OBJECTID&resultOffset=4000&resultRecordCount=2000",
            "sha256": "59961bdc25052d9a343643735da35609d677faf26194c2fd147eabcbc0e52d01",
            "bytes": 510843,
            "features": 2000,
        },
        {
            "file": "mbhg-2026-regc-p003.json",
            "url": "https://services2.arcgis.com/vKb0s8tBIA3bdocZ/arcgis/rest/services/Meshblock_Higher_Geographies_2026/FeatureServer/0/query?where=1%3D1&outFields=MB2026_V1_00,SA12026_V1_00,TA2026_V1_00,TA2026_V1_00_NAME,REGC2025_V1_00,REGC2025_V1_00_NAME,TALB2026_V1_00,TALB2026_V1_00_NAME&returnGeometry=false&f=json&orderByFields=OBJECTID&resultOffset=6000&resultRecordCount=2000",
            "sha256": "7ed90527fba6c68a523e6db150e4cb129978a35676f8201a26f12e4533303ec2",
            "bytes": 507377,
            "features": 2000,
        },
        {
            "file": "mbhg-2026-regc-p004.json",
            "url": "https://services2.arcgis.com/vKb0s8tBIA3bdocZ/arcgis/rest/services/Meshblock_Higher_Geographies_2026/FeatureServer/0/query?where=1%3D1&outFields=MB2026_V1_00,SA12026_V1_00,TA2026_V1_00,TA2026_V1_00_NAME,REGC2025_V1_00,REGC2025_V1_00_NAME,TALB2026_V1_00,TALB2026_V1_00_NAME&returnGeometry=false&f=json&orderByFields=OBJECTID&resultOffset=8000&resultRecordCount=2000",
            "sha256": "b7f1e845c837124b1a07e54b2c8b815c225a00deacc75f237b944e5e7d064f39",
            "bytes": 510956,
            "features": 2000,
        },
        {
            "file": "mbhg-2026-regc-p005.json",
            "url": "https://services2.arcgis.com/vKb0s8tBIA3bdocZ/arcgis/rest/services/Meshblock_Higher_Geographies_2026/FeatureServer/0/query?where=1%3D1&outFields=MB2026_V1_00,SA12026_V1_00,TA2026_V1_00,TA2026_V1_00_NAME,REGC2025_V1_00,REGC2025_V1_00_NAME,TALB2026_V1_00,TALB2026_V1_00_NAME&returnGeometry=false&f=json&orderByFields=OBJECTID&resultOffset=10000&resultRecordCount=2000",
            "sha256": "a6e6514c2ac7537ab2d0b0158c200de564167b1599f80d7a7d570070b542e9dd",
            "bytes": 508691,
            "features": 2000,
        },
        {
            "file": "mbhg-2026-regc-p006.json",
            "url": "https://services2.arcgis.com/vKb0s8tBIA3bdocZ/arcgis/rest/services/Meshblock_Higher_Geographies_2026/FeatureServer/0/query?where=1%3D1&outFields=MB2026_V1_00,SA12026_V1_00,TA2026_V1_00,TA2026_V1_00_NAME,REGC2025_V1_00,REGC2025_V1_00_NAME,TALB2026_V1_00,TALB2026_V1_00_NAME&returnGeometry=false&f=json&orderByFields=OBJECTID&resultOffset=12000&resultRecordCount=2000",
            "sha256": "ce785717911faccfab894dd49e9c2f692e73d42cee62909a2cc779d9db06ed1c",
            "bytes": 506340,
            "features": 2000,
        },
        {
            "file": "mbhg-2026-regc-p007.json",
            "url": "https://services2.arcgis.com/vKb0s8tBIA3bdocZ/arcgis/rest/services/Meshblock_Higher_Geographies_2026/FeatureServer/0/query?where=1%3D1&outFields=MB2026_V1_00,SA12026_V1_00,TA2026_V1_00,TA2026_V1_00_NAME,REGC2025_V1_00,REGC2025_V1_00_NAME,TALB2026_V1_00,TALB2026_V1_00_NAME&returnGeometry=false&f=json&orderByFields=OBJECTID&resultOffset=14000&resultRecordCount=2000",
            "sha256": "ebcdf665068b37447bcddb629bc4d2f9bdff099113eae506fd922089f5b20aa8",
            "bytes": 535730,
            "features": 2000,
        },
        {
            "file": "mbhg-2026-regc-p008.json",
            "url": "https://services2.arcgis.com/vKb0s8tBIA3bdocZ/arcgis/rest/services/Meshblock_Higher_Geographies_2026/FeatureServer/0/query?where=1%3D1&outFields=MB2026_V1_00,SA12026_V1_00,TA2026_V1_00,TA2026_V1_00_NAME,REGC2025_V1_00,REGC2025_V1_00_NAME,TALB2026_V1_00,TALB2026_V1_00_NAME&returnGeometry=false&f=json&orderByFields=OBJECTID&resultOffset=16000&resultRecordCount=2000",
            "sha256": "ee2f315e8979e745d6ed5c4b0c99901839366a793b4d9abcc9fd68c5897199c0",
            "bytes": 528114,
            "features": 2000,
        },
        {
            "file": "mbhg-2026-regc-p009.json",
            "url": "https://services2.arcgis.com/vKb0s8tBIA3bdocZ/arcgis/rest/services/Meshblock_Higher_Geographies_2026/FeatureServer/0/query?where=1%3D1&outFields=MB2026_V1_00,SA12026_V1_00,TA2026_V1_00,TA2026_V1_00_NAME,REGC2025_V1_00,REGC2025_V1_00_NAME,TALB2026_V1_00,TALB2026_V1_00_NAME&returnGeometry=false&f=json&orderByFields=OBJECTID&resultOffset=18000&resultRecordCount=2000",
            "sha256": "06ccf71f83243bbae575b8055a89e08333bc78a5a1e2f509dab4251a7e05b7cf",
            "bytes": 518165,
            "features": 2000,
        },
        {
            "file": "mbhg-2026-regc-p010.json",
            "url": "https://services2.arcgis.com/vKb0s8tBIA3bdocZ/arcgis/rest/services/Meshblock_Higher_Geographies_2026/FeatureServer/0/query?where=1%3D1&outFields=MB2026_V1_00,SA12026_V1_00,TA2026_V1_00,TA2026_V1_00_NAME,REGC2025_V1_00,REGC2025_V1_00_NAME,TALB2026_V1_00,TALB2026_V1_00_NAME&returnGeometry=false&f=json&orderByFields=OBJECTID&resultOffset=20000&resultRecordCount=2000",
            "sha256": "2e54c88fa2c53a4131605adf1a25268df62de4c9587164e56c669187f0b45792",
            "bytes": 530076,
            "features": 2000,
        },
        {
            "file": "mbhg-2026-regc-p011.json",
            "url": "https://services2.arcgis.com/vKb0s8tBIA3bdocZ/arcgis/rest/services/Meshblock_Higher_Geographies_2026/FeatureServer/0/query?where=1%3D1&outFields=MB2026_V1_00,SA12026_V1_00,TA2026_V1_00,TA2026_V1_00_NAME,REGC2025_V1_00,REGC2025_V1_00_NAME,TALB2026_V1_00,TALB2026_V1_00_NAME&returnGeometry=false&f=json&orderByFields=OBJECTID&resultOffset=22000&resultRecordCount=2000",
            "sha256": "685e71a8b26fe2d33da66cadb350ab6de65daca0a4397bcc5699e4730dab5bfb",
            "bytes": 542506,
            "features": 2000,
        },
        {
            "file": "mbhg-2026-regc-p012.json",
            "url": "https://services2.arcgis.com/vKb0s8tBIA3bdocZ/arcgis/rest/services/Meshblock_Higher_Geographies_2026/FeatureServer/0/query?where=1%3D1&outFields=MB2026_V1_00,SA12026_V1_00,TA2026_V1_00,TA2026_V1_00_NAME,REGC2025_V1_00,REGC2025_V1_00_NAME,TALB2026_V1_00,TALB2026_V1_00_NAME&returnGeometry=false&f=json&orderByFields=OBJECTID&resultOffset=24000&resultRecordCount=2000",
            "sha256": "a0829709d98edea532a5b8fd19371734e7a374e01341edc232ed62102dd8163c",
            "bytes": 535178,
            "features": 2000,
        },
        {
            "file": "mbhg-2026-regc-p013.json",
            "url": "https://services2.arcgis.com/vKb0s8tBIA3bdocZ/arcgis/rest/services/Meshblock_Higher_Geographies_2026/FeatureServer/0/query?where=1%3D1&outFields=MB2026_V1_00,SA12026_V1_00,TA2026_V1_00,TA2026_V1_00_NAME,REGC2025_V1_00,REGC2025_V1_00_NAME,TALB2026_V1_00,TALB2026_V1_00_NAME&returnGeometry=false&f=json&orderByFields=OBJECTID&resultOffset=26000&resultRecordCount=2000",
            "sha256": "41d4db88e39694cc4354209f1bfff710f07f0dcd86aeb6f2d2830a025f803d4e",
            "bytes": 514352,
            "features": 2000,
        },
        {
            "file": "mbhg-2026-regc-p014.json",
            "url": "https://services2.arcgis.com/vKb0s8tBIA3bdocZ/arcgis/rest/services/Meshblock_Higher_Geographies_2026/FeatureServer/0/query?where=1%3D1&outFields=MB2026_V1_00,SA12026_V1_00,TA2026_V1_00,TA2026_V1_00_NAME,REGC2025_V1_00,REGC2025_V1_00_NAME,TALB2026_V1_00,TALB2026_V1_00_NAME&returnGeometry=false&f=json&orderByFields=OBJECTID&resultOffset=28000&resultRecordCount=2000",
            "sha256": "b577cce44704864fc6bba62e088f766af70f8f5fc0725c7d2ee2afda6fa968e2",
            "bytes": 523320,
            "features": 2000,
        },
        {
            "file": "mbhg-2026-regc-p015.json",
            "url": "https://services2.arcgis.com/vKb0s8tBIA3bdocZ/arcgis/rest/services/Meshblock_Higher_Geographies_2026/FeatureServer/0/query?where=1%3D1&outFields=MB2026_V1_00,SA12026_V1_00,TA2026_V1_00,TA2026_V1_00_NAME,REGC2025_V1_00,REGC2025_V1_00_NAME,TALB2026_V1_00,TALB2026_V1_00_NAME&returnGeometry=false&f=json&orderByFields=OBJECTID&resultOffset=30000&resultRecordCount=2000",
            "sha256": "aa5d2575c93d841a6d2119cad8bb18b41c681d7c94766f547733608004be53ff",
            "bytes": 513997,
            "features": 2000,
        },
        {
            "file": "mbhg-2026-regc-p016.json",
            "url": "https://services2.arcgis.com/vKb0s8tBIA3bdocZ/arcgis/rest/services/Meshblock_Higher_Geographies_2026/FeatureServer/0/query?where=1%3D1&outFields=MB2026_V1_00,SA12026_V1_00,TA2026_V1_00,TA2026_V1_00_NAME,REGC2025_V1_00,REGC2025_V1_00_NAME,TALB2026_V1_00,TALB2026_V1_00_NAME&returnGeometry=false&f=json&orderByFields=OBJECTID&resultOffset=32000&resultRecordCount=2000",
            "sha256": "a2841c24d03b085993c6537bbe3d741275cc85b7f4c637469e193c39bbfc970d",
            "bytes": 522196,
            "features": 2000,
        },
        {
            "file": "mbhg-2026-regc-p017.json",
            "url": "https://services2.arcgis.com/vKb0s8tBIA3bdocZ/arcgis/rest/services/Meshblock_Higher_Geographies_2026/FeatureServer/0/query?where=1%3D1&outFields=MB2026_V1_00,SA12026_V1_00,TA2026_V1_00,TA2026_V1_00_NAME,REGC2025_V1_00,REGC2025_V1_00_NAME,TALB2026_V1_00,TALB2026_V1_00_NAME&returnGeometry=false&f=json&orderByFields=OBJECTID&resultOffset=34000&resultRecordCount=2000",
            "sha256": "1e96c882a4e0a62665b88258e580790fc270f96bd30fb24fdc3c1057a005b01c",
            "bytes": 521764,
            "features": 2000,
        },
        {
            "file": "mbhg-2026-regc-p018.json",
            "url": "https://services2.arcgis.com/vKb0s8tBIA3bdocZ/arcgis/rest/services/Meshblock_Higher_Geographies_2026/FeatureServer/0/query?where=1%3D1&outFields=MB2026_V1_00,SA12026_V1_00,TA2026_V1_00,TA2026_V1_00_NAME,REGC2025_V1_00,REGC2025_V1_00_NAME,TALB2026_V1_00,TALB2026_V1_00_NAME&returnGeometry=false&f=json&orderByFields=OBJECTID&resultOffset=36000&resultRecordCount=2000",
            "sha256": "cc7521961e6c58b76fde38b0f9d733b2cc548320507a3190dd7dd8064ed2a3cd",
            "bytes": 501252,
            "features": 2000,
        },
        {
            "file": "mbhg-2026-regc-p019.json",
            "url": "https://services2.arcgis.com/vKb0s8tBIA3bdocZ/arcgis/rest/services/Meshblock_Higher_Geographies_2026/FeatureServer/0/query?where=1%3D1&outFields=MB2026_V1_00,SA12026_V1_00,TA2026_V1_00,TA2026_V1_00_NAME,REGC2025_V1_00,REGC2025_V1_00_NAME,TALB2026_V1_00,TALB2026_V1_00_NAME&returnGeometry=false&f=json&orderByFields=OBJECTID&resultOffset=38000&resultRecordCount=2000",
            "sha256": "21a908f590b627511552269556877575ebc07739e8ffcf5193b35f84c57f9e82",
            "bytes": 520360,
            "features": 2000,
        },
        {
            "file": "mbhg-2026-regc-p020.json",
            "url": "https://services2.arcgis.com/vKb0s8tBIA3bdocZ/arcgis/rest/services/Meshblock_Higher_Geographies_2026/FeatureServer/0/query?where=1%3D1&outFields=MB2026_V1_00,SA12026_V1_00,TA2026_V1_00,TA2026_V1_00_NAME,REGC2025_V1_00,REGC2025_V1_00_NAME,TALB2026_V1_00,TALB2026_V1_00_NAME&returnGeometry=false&f=json&orderByFields=OBJECTID&resultOffset=40000&resultRecordCount=2000",
            "sha256": "5fa0fb56d80928156336393b16c2ded43723b2451a3beaa0bebf4405c4966e95",
            "bytes": 521326,
            "features": 2000,
        },
        {
            "file": "mbhg-2026-regc-p021.json",
            "url": "https://services2.arcgis.com/vKb0s8tBIA3bdocZ/arcgis/rest/services/Meshblock_Higher_Geographies_2026/FeatureServer/0/query?where=1%3D1&outFields=MB2026_V1_00,SA12026_V1_00,TA2026_V1_00,TA2026_V1_00_NAME,REGC2025_V1_00,REGC2025_V1_00_NAME,TALB2026_V1_00,TALB2026_V1_00_NAME&returnGeometry=false&f=json&orderByFields=OBJECTID&resultOffset=42000&resultRecordCount=2000",
            "sha256": "1687f3810ac3e6d102cfcfba46a4a9144783d9e25dd53f65bfe5181a054eb756",
            "bytes": 514036,
            "features": 2000,
        },
        {
            "file": "mbhg-2026-regc-p022.json",
            "url": "https://services2.arcgis.com/vKb0s8tBIA3bdocZ/arcgis/rest/services/Meshblock_Higher_Geographies_2026/FeatureServer/0/query?where=1%3D1&outFields=MB2026_V1_00,SA12026_V1_00,TA2026_V1_00,TA2026_V1_00_NAME,REGC2025_V1_00,REGC2025_V1_00_NAME,TALB2026_V1_00,TALB2026_V1_00_NAME&returnGeometry=false&f=json&orderByFields=OBJECTID&resultOffset=44000&resultRecordCount=2000",
            "sha256": "c86bdd6358ca59ebbd649d999757421225f8e6765ed9c77dc2068dfac76d93fa",
            "bytes": 518806,
            "features": 2000,
        },
        {
            "file": "mbhg-2026-regc-p023.json",
            "url": "https://services2.arcgis.com/vKb0s8tBIA3bdocZ/arcgis/rest/services/Meshblock_Higher_Geographies_2026/FeatureServer/0/query?where=1%3D1&outFields=MB2026_V1_00,SA12026_V1_00,TA2026_V1_00,TA2026_V1_00_NAME,REGC2025_V1_00,REGC2025_V1_00_NAME,TALB2026_V1_00,TALB2026_V1_00_NAME&returnGeometry=false&f=json&orderByFields=OBJECTID&resultOffset=46000&resultRecordCount=2000",
            "sha256": "05b77539c17a8372244c5d32f198e5fccd5627283056df86edebbc33422996aa",
            "bytes": 520836,
            "features": 2000,
        },
        {
            "file": "mbhg-2026-regc-p024.json",
            "url": "https://services2.arcgis.com/vKb0s8tBIA3bdocZ/arcgis/rest/services/Meshblock_Higher_Geographies_2026/FeatureServer/0/query?where=1%3D1&outFields=MB2026_V1_00,SA12026_V1_00,TA2026_V1_00,TA2026_V1_00_NAME,REGC2025_V1_00,REGC2025_V1_00_NAME,TALB2026_V1_00,TALB2026_V1_00_NAME&returnGeometry=false&f=json&orderByFields=OBJECTID&resultOffset=48000&resultRecordCount=2000",
            "sha256": "1677d482d346196e854dddea8b392d64f6966b96178ecaaf784ac561f65102f2",
            "bytes": 521491,
            "features": 2000,
        },
        {
            "file": "mbhg-2026-regc-p025.json",
            "url": "https://services2.arcgis.com/vKb0s8tBIA3bdocZ/arcgis/rest/services/Meshblock_Higher_Geographies_2026/FeatureServer/0/query?where=1%3D1&outFields=MB2026_V1_00,SA12026_V1_00,TA2026_V1_00,TA2026_V1_00_NAME,REGC2025_V1_00,REGC2025_V1_00_NAME,TALB2026_V1_00,TALB2026_V1_00_NAME&returnGeometry=false&f=json&orderByFields=OBJECTID&resultOffset=50000&resultRecordCount=2000",
            "sha256": "b84a06ee52108d764dbcf2f952bfbae3a7b8bd78ca715f87e5ae4ca88115e1e7",
            "bytes": 520123,
            "features": 2000,
        },
        {
            "file": "mbhg-2026-regc-p026.json",
            "url": "https://services2.arcgis.com/vKb0s8tBIA3bdocZ/arcgis/rest/services/Meshblock_Higher_Geographies_2026/FeatureServer/0/query?where=1%3D1&outFields=MB2026_V1_00,SA12026_V1_00,TA2026_V1_00,TA2026_V1_00_NAME,REGC2025_V1_00,REGC2025_V1_00_NAME,TALB2026_V1_00,TALB2026_V1_00_NAME&returnGeometry=false&f=json&orderByFields=OBJECTID&resultOffset=52000&resultRecordCount=2000",
            "sha256": "e10e772428ed3a9dbbb2c5f855c56d64503eb00da5a8b4201a1ca148a43ab4b7",
            "bytes": 517802,
            "features": 2000,
        },
        {
            "file": "mbhg-2026-regc-p027.json",
            "url": "https://services2.arcgis.com/vKb0s8tBIA3bdocZ/arcgis/rest/services/Meshblock_Higher_Geographies_2026/FeatureServer/0/query?where=1%3D1&outFields=MB2026_V1_00,SA12026_V1_00,TA2026_V1_00,TA2026_V1_00_NAME,REGC2025_V1_00,REGC2025_V1_00_NAME,TALB2026_V1_00,TALB2026_V1_00_NAME&returnGeometry=false&f=json&orderByFields=OBJECTID&resultOffset=54000&resultRecordCount=2000",
            "sha256": "ebe3c768a5caa52a56a3a3ad4c0b2a2e060aa1839aa5a4f88bbd8a4bc07bf57e",
            "bytes": 518919,
            "features": 2000,
        },
        {
            "file": "mbhg-2026-regc-p028.json",
            "url": "https://services2.arcgis.com/vKb0s8tBIA3bdocZ/arcgis/rest/services/Meshblock_Higher_Geographies_2026/FeatureServer/0/query?where=1%3D1&outFields=MB2026_V1_00,SA12026_V1_00,TA2026_V1_00,TA2026_V1_00_NAME,REGC2025_V1_00,REGC2025_V1_00_NAME,TALB2026_V1_00,TALB2026_V1_00_NAME&returnGeometry=false&f=json&orderByFields=OBJECTID&resultOffset=56000&resultRecordCount=2000",
            "sha256": "a31cdd9e44109207e0e55e09ec515014e97b0b6dccb362578854e7eb57c0b05c",
            "bytes": 409787,
            "features": 1575,
        },
    ],
    "layers": [],
}


def metadata_url(query_url):
    """Return the layer metadata endpoint for a FeatureServer query URL."""
    if not query_url.endswith("/query"):
        raise ValueError(f"Expected a FeatureServer query URL: {query_url}")
    return query_url[: -len("/query")] + "?f=json"


def pinned_downloads(provenance):
    """List (file, url, sha256, bytes-or-None) for every pinned input."""
    downloads = [
        (page["file"], page["url"], page["sha256"], page.get("bytes"))
        for page in provenance["pages"]
    ]
    downloads.extend(
        (
            layer["metadata_file"],
            metadata_url(layer["query_url"]),
            layer["metadata_sha256"],
            None,
        )
        for layer in provenance["layers"]
    )
    names = [name for name, *_ in downloads]
    if len(names) != len(set(names)):
        raise ValueError("Pinned file names must be unique")
    return downloads


def verify(name, payload, sha256, size):
    """Refuse bytes that differ from their pin."""
    if size is not None and len(payload) != size:
        raise ValueError(f"{name}: {len(payload)} bytes, pinned {size}")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != sha256:
        raise ValueError(f"{name}: SHA-256 {digest}, pinned {sha256}")


def _fetch(url, attempts=4):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return response.read()
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(10 * (attempt + 1))
    raise AssertionError("unreachable")


def fetch_all(provenance, out_dir, fetch=_fetch):
    """Fetch every pinned input into out_dir; keep files that already verify."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fetched = []
    for name, url, sha256, size in pinned_downloads(provenance):
        target = out_dir / name
        if target.exists():
            try:
                verify(name, target.read_bytes(), sha256, size)
                continue
            except ValueError:
                pass
        payload = fetch(url)
        verify(name, payload, sha256, size)
        target.write_bytes(payload)
        fetched.append(name)
    return fetched


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--spec", type=Path, default=SPEC)
    selector.add_argument("--joint-geography", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    source_env = (
        "NZ_JOINT_GEOGRAPHY_SOURCES" if args.joint_geography else "NZ_AS_AREA_SOURCES"
    )
    out_dir = args.out or os.environ.get(source_env)
    if out_dir is None:
        parser.error(f"supply --out DIR or {source_env}=DIR")
    out_dir = Path(out_dir)
    provenance = (
        JOINT_GEOGRAPHY_PROVENANCE
        if args.joint_geography
        else json.loads(args.spec.read_text())["provenance"]
    )
    fetched = fetch_all(provenance, out_dir)
    total = len(pinned_downloads(provenance))
    print(f"{total} pinned inputs verified in {out_dir}; {len(fetched)} fetched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
