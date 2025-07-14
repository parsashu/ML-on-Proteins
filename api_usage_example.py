import requests
import json


def test_health_endpoint():
    """Test the health endpoint to verify API is running"""
    url = "https://stable-protein.subnitro.pro/health"

    print("🔍 Testing API Health...")
    try:
        response = requests.get(url, timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def improve_protein_stability(protein_data):
    """Send a request to improve protein stability"""
    url = "https://stable-protein.subnitro.pro/improve-stability"
    headers = {"Content-Type": "application/json"}

    print(f"\n🚀 Making request to: {url}")
    print(f"📦 Payload:")
    print(json.dumps(protein_data, indent=2))
    print("\n⏳ Processing 371 protein mutations (this may take 1-2 minutes)...")

    try:
        response = requests.post(
            url, headers=headers, json=protein_data, timeout=None
        )  # Infinite timeout
        print(f"\n📊 Response Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                data = result["data"]
                print("✅ SUCCESS! Protein stability improvement found!")
                print("=" * 50)
                print(f"🔬 Original Sequence: {protein_data['Protein_Sequence']}")
                print(
                    f"🎯 Best Mutation: {data['original_amino_acid']}{data['changed_position']}{data['changed_amino_acid']}"
                )
                print(f"📈 Tm Improvement: +{data['tm_change']:.2f}°C")
                print(f"🔄 Changed: {data['is_changed']}")
                if data["is_changed"]:
                    print(f"🧬 New Sequence: {data['best_sequence']}")
                print("=" * 50)
                return data
            else:
                print(f"❌ API Error: {result.get('error')}")
                return None
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return None

    except requests.exceptions.Timeout:
        print("⏰ Request timed out after 3 minutes")
        print(
            "💡 The API is processing 371 mutations. Try again or contact support if this persists."
        )
        return None
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None


def main():
    """Main function demonstrating API usage"""
    print("🧬 Protein Stability Improvement API")
    print("=" * 50)

    # Test health endpoint first
    if not test_health_endpoint():
        print("❌ API is not available. Please check if the server is running.")
        return

    print("\n✅ API is healthy and ready!")

    # Example protein data
    protein_data = {
        "ASA": 9.0,  # Accessible Surface Area
        "pH": 6.0,  # pH value
        "Tm_(C)": 80.4,  # Melting temperature in Celsius
        "Coil": 1,  # Binary indicator for coil structure
        "Helix": 0,  # Binary indicator for helix structure
        "Sheet": 0,  # Binary indicator for sheet structure
        "Turn": 0,  # Binary indicator for turn structure
        "Protein_Sequence": "MGDVEKGKKIFVQKCAQCHTVEKGGKHKTGPNLHGLFGRKTGQAPGFTYTDANKNKGITWKEETLMEYLENPKKYIPGTKMIFAGIKKKTEREDLIAYLKKATNE",
    }

    # Improve protein stability
    result = improve_protein_stability(protein_data)

    if result:
        print("\n📋 Summary:")
        print(f"• The API analyzed {len(protein_data['Protein_Sequence'])} amino acids")
        print(f"• Generated {371} different mutations (original + 370 variants)")
        print(
            f"• Found the best mutation to improve stability by {result['tm_change']:.2f}°C"
        )
        print(
            f"• Recommended change: {result['original_amino_acid']} → {result['changed_amino_acid']} at position {result['changed_position']}"
        )


if __name__ == "__main__":
    main()
