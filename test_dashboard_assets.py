import os
import json

def verify_dashboard():
    dashboard_dir = "dashboard"
    required_files = [
        "index.html",
        "app.js",
        "styles.css",
        "lib/three.module.js",
        "lib/three.core.js",
        "lib/OrbitControls.js",
        "data/patients_data.json",
        "data/global_importance.json"
    ]
    
    print("Verifying dashboard asset existence:")
    for rel_path in required_files:
        full_path = os.path.join(dashboard_dir, rel_path)
        exists = os.path.exists(full_path)
        size = os.path.getsize(full_path) if exists else 0
        print(f"  [{'PASS' if exists else 'FAIL'}] {rel_path} ({size} bytes)")
        assert exists, f"Missing file: {full_path}"
        assert size > 0, f"Empty file: {full_path}"

    with open(os.path.join(dashboard_dir, "data/patients_data.json"), "r") as f:
        data = json.load(f)
        assert len(data) >= 5, "Expected at least 5 patient profiles"
        for p in data:
            assert "predictions" in p
            assert "measurements" in p
            assert "shap_contributions" in p
            assert "CAD" in p["predictions"]
            assert "LAD" in p["predictions"]
            assert "LCX" in p["predictions"]
            assert "RCA" in p["predictions"]
            print(f"  Verified profile: {p['title']} (CAD Prob: {p['predictions']['CAD']['probability']}%)")

    print("\nALL DASHBOARD ASSET CHECKS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    verify_dashboard()
