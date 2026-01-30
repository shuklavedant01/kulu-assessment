# Project Report: Cutout Detection Analysis

## 4. Cutout Detection: How It Works 🔍

The detector uses a set of **Contextual Rules** to distinguish between natural pauses and technical cutouts. It doesn't just look for "silence" (because a pause for thinking is normal).

### The Rules of Detection

| Rule | Name | Condition / Logic |
| :--- | :--- | :--- |
| **1** | **Incomplete Speech** | Did a sentence end abruptly without punctuation? <br> *Checks for words like "and", "but", "so" or lack of punctuation.* |
| **2** | **Missing Response** | Did the User ask a question, and Agent failed to respond? <br> *Checks for question marks or question words followed by silence.* |
| **3** | **Mid-Speech Gap** | Did the same speaker pause for > 2s in the middle of a phrase? |
| **4** | **Within-Sentence Dropout** | Tiny gaps (0.7s - 2s) inside a continuous speech segment. <br> *Usually indicates packet loss or technical glitch.* |
| **5** | **User Drop-off** | **Did the user speak for < 1.0s and then vanish?** <br> *Indicates potential disconnection or microphone failure immediately after starting to speak.* |
| **6** | **Agent Latency** | Gaps > 3s before Agent response (not strictly a cutout, but flagged). |

---

### 🧮 Detailed Calculations

Here is the exact math used for the thresholds, including the "User Drop-off" rule you asked about.

#### 1. User Drop-off Calculation
*   **Target**: Trigger when a user starts speaking but cuts out almost immediately.
*   **Formula**:
    $$ Cutout = (D_{speech} < 1.0s) \land (D_{gap} > 5.0s) $$
    *   **$D_{speech}$ (Speech Duration)**: The duration of the user's audio segment before the silence.
        *   *Old Value*: 0.5s
        *   **New Value**: **1.0s** (Updated to be more sensitive)
    *   **$D_{gap}$ (Silence Duration)**: The duration of the silence immediately following the speech.
    *   **Logic**: If a user says something very short (less than 1 second) and then silence follows for more than 5 seconds, it is flagged as a "User Drop-off".

#### 2. Adaptive Thresholds (Smart Detection)
For other rules, we calculate a baseline from the conversation itself to adapt to the speaker's pace.
*   **Median Response Time ($M$)**: The median time the Agent usually takes to respond.
*   **P95 Response Time ($P_{95}$)**: The 95th percentile (slowest 5%) of responses.

**Missing Response Threshold**:
$$ T_{missing} = \max(M \times 2.5, \ P_{95} \times 1.5, \ 3.0s) $$
*   We flag a missing response only if the silence is **2.5x longer** than the median response time.

**Incomplete Speech Threshold**:
$$ T_{incomplete} = \max(M_{pause} \times 1.5, \ P_{90\_pause} \times 1.2, \ 0.8s) $$
*   We flag a pause after incomplete syntax (e.g., "I went to...") if it is **1.5x longer** than their normal pause duration.
