<b>Goal: get help without leaving easyROB.</b><br><br>Click <b>robBOT</b> at the top right of the main window. You can ask about GUI controls, workflow stages, results, warnings, and AQME/ROBERT documentation while the main interface stays open.

---

<b>Recognize the two tabs.</b><br><br>robBOT opens in a separate window. <b>Chat</b> contains the conversation and question box; <b>Settings</b> controls how answers are produced. The first Chat message explains the modes. Use <b>Close</b> to hide the window and the robBOT button to reopen it.

---

<b>Ask a specific question.</b><br><br>Type in the input box and press <b>Send</b> or Enter. For example, ask “What does VERIFY check in my workflow?” Mention the step or result you want explained. <b>Clear</b> resets the visible conversation when you want to start again.

---

<b>Use the built-in prompts.</b><br><br>Expand <b>Advanced options</b> and choose a <b>Suggested question</b>. This fills the question box; press Send to receive the answer. The same area shows token usage and, when an answer provides them, source links. Questions about Check ML and robBOT are included.

---

<b>Summarize detected work.</b><br><br>After opening a project with outputs, Chat may offer <b>Summarize workflow</b> or <b>Summarize report</b>. Click it for an explanation based on the detected results. If no result is available, the button stays disabled. The nearby status tells you what kind of output robBOT found.

---

<b>Choose the answer mode.</b><br><br>Open <b>Settings</b> and use the <b>Mode</b> dropdown. <b>Cloud AI</b> uses a provider and API key, <b>Local AI</b> uses a model in your ROBERT environment, and <b>Heuristic</b> gives rule-based help from local context. The settings below change with the mode.

---

<b>Configure Cloud AI if you want it.</b><br><br>Select a provider, enter its API key, and click <b>Save API</b>. The key field is masked. The optional web-search checkbox allows supported cloud providers to look up missing or current information; provider use and searches may incur charges. The on-screen guide explains the Groq setup.

---

<b>Prepare Local AI if preferred.</b><br><br>Select <b>Local AI</b> to see whether its model is ready, downloaded, or still missing. Use the action button to download or load it when needed. Once ready, return to Chat and ask normally; this mode does not need a cloud provider key.

---

<b>Use Heuristic for local guidance.</b><br><br>Select <b>Heuristic</b> when you want a rule-based answer using the GUI state, popups, logs, and packaged knowledge. It needs no API key or AI model. Return to Chat, ask a focused question, and inspect the answer alongside the relevant GUI output.
