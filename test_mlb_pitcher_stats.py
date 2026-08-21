import json
from urllib.request import urlopen, Request

YEAR = 2026

# Sandy Alcantara from the Baseball Savant test
PLAYER_ID = 645261


def get_json(url):
    request = Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    with urlopen(request, timeout=20) as response:
        return json.load(response)


url = (
    f"https://statsapi.mlb.com/api/v1/"
    f"people/{PLAYER_ID}/stats"
    f"?stats=season"
    f"&group=pitching"
    f"&season={YEAR}"
)

print("Connecting to MLB pitcher stats...")
print()

data = get_json(url)

try:
    stat = data["stats"][0]["splits"][0]["stat"]

    strikeouts = stat.get("strikeOuts", 0)
    walks = stat.get("baseOnBalls", 0)
    batters_faced = stat.get("battersFaced", 0)

    if batters_faced > 0:
        k_percent = (strikeouts / batters_faced) * 100
        bb_percent = (walks / batters_faced) * 100
    else:
        k_percent = 0
        bb_percent = 0

    output = []

    output.append("SHARPREPORT MLB PITCHER TEST")
    output.append("=" * 45)
    output.append("")
    output.append(f"Player ID: {PLAYER_ID}")
    output.append(f"Strikeouts: {strikeouts}")
    output.append(f"Walks: {walks}")
    output.append(f"Batters Faced: {batters_faced}")
    output.append("")
    output.append(f"K%: {k_percent:.1f}%")
    output.append(f"BB%: {bb_percent:.1f}%")
    output.append("")
    output.append("ALL AVAILABLE STAT FIELDS:")
    output.append("")

    for key, value in stat.items():
        output.append(f"{key}: {value}")

    with open(
        "mlb_pitcher_stats_output.txt",
        "w",
        encoding="utf-8"
    ) as file:
        file.write("\n".join(output))

    print("TEST COMPLETE")
    print()
    print("Open mlb_pitcher_stats_output.txt")

except Exception as error:

    print("ERROR:")
    print(error)

    print()
    print("RAW RESPONSE:")
    print(data)