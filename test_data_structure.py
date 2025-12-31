import seispolarity.data
import inspect
from seispolarity import config

print("Testing SeisPolarity Data Module...")

# 1. Check imports
datasets = seispolarity.data.__all__
print(f"Found {len(datasets)} exported symbols in seispolarity.data")

# 2. Check classes
for name in datasets:
    try:
        cls = getattr(seispolarity.data, name)
        if inspect.isclass(cls):
            # Check if it inherits from WaveformBenchmarkDataset or MultiWaveformDataset
            bases = [b.__name__ for b in cls.__mro__]
            if "WaveformBenchmarkDataset" in bases or "MultiWaveformDataset" in bases:
                 print(f"✅ {name} is a valid dataset class")
            elif name in ["WaveformBenchmarkDataset", "MultiWaveformDataset"]:
                 print(f"✅ {name} is a base class")
            else:
                 print(f"⚠️ {name} is a class but might not be a dataset ({bases})")
        else:
            print(f"⚠️ {name} is not a class ({type(cls)})")
    except Exception as e:
        print(f"❌ Failed to inspect {name}: {e}")

# 3. Check config
print(f"\nRemote Root: {config.Settings.remote_root}")

print("\nTest Complete.")
