from cutout_detector_adaptive import detect_user_drop_off

# Test Case 1: Drop-off (Should detect)
# User speaks for 0.8s (briefly)
# Silence follows for 6.0s (vanished)
gap1 = {
    'before_speaker': 'User',
    'before_duration': 0.8,
    'duration': 6.0
}

# Test Case 2: Natural Pause (Should NOT detect)
# User speaks for 2.0s (normal speech)
# Silence follows for 6.0s
gap2 = {
    'before_speaker': 'User',
    'before_duration': 2.0,
    'duration': 6.0
}

# Test Case 3: Brief Silence (Should NOT detect)
# User speaks for 0.8s
# Silence follows for 2.0s (just a pause)
gap3 = {
    'before_speaker': 'User',
    'before_duration': 0.8,
    'duration': 2.0
}

def run_test(name, gap):
    result = detect_user_drop_off(gap)
    status = "DETECTED" if result else "NOT DETECTED"
    print(f"{name}: {status}")
    if result:
        print(f"  Reason: {result['reason']}")

print("Running Unit Tests for User Drop-off Rule (Threshold: <1.0s speech, >5.0s gap)\n")
run_test("Test 1 (True Positive)", gap1)  # Expect DETECTED
run_test("Test 2 (Long Speech)", gap2)    # Expect NOT DETECTED
run_test("Test 3 (Short Gap)", gap3)      # Expect NOT DETECTED
