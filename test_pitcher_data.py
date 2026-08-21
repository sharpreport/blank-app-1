from pybaseball import statcast_pitcher_expected_stats

YEAR = 2026

print("Connecting to Baseball Savant...")
print()

data = statcast_pitcher_expected_stats(
    YEAR,
    minPA=1
)

print("PITCHER DATA READY")
print()

print("Number of pitchers returned:")
print(len(data))

print()
print("COLUMN NAMES:")
print()

for column in data.columns:
    print(column)

print()
print("FIRST 5 PITCHERS:")
print()

print(data.head(5).to_string())