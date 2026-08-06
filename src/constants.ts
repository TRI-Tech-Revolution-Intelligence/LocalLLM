import type { AgentPermissions, EvalBenchmarkType, ServerConfig } from "./types";

export const preserveThinkingDefault = '{"preserve_thinking": true}';
export const themeStorageKey = "localllm-theme";
export const paneWidthStorageKey = "localllm-left-pane-width";
export const benchmarkSettingsStorageKey = "localllm-benchmark-settings";
export const agentPermissionsStorageKey = "localllm-agent-permissions";
export const agentYoloModeStorageKey = "localllm-agent-yolo-mode";
export const agentAutoAcceptStorageKey = "localllm-agent-auto-accept";
export const agentGoalStorageKey = "localllm-agent-goal";
export const agentModeStorageKey = "localllm-agent-mode";
export const agentExecutionModeStorageKey = "localllm-agent-execution-mode";
export const agentActiveProfileStorageKey = "localllm-agent-active-profile";
export const agentThinkingLevelStorageKey = "localllm-agent-thinking-level";
export const agentTodosStorageKey = "localllm-agent-todos";
export const agentSkillsStorageKey = "localllm-agent-skills";
export const agentSubagentsStorageKey = "localllm-agent-subagents";
export const agentAutoCompactStorageKey = "localllm-agent-auto-compact";
export const agentContextSummaryStorageKey = "localllm-agent-context-summary";
export const agentMcpServersStorageKey = "localllm-agent-mcp-servers";
export const agentTaskHistoryStorageKey = "localllm-agent-task-history";
export const agentProfilesStorageKey = "localllm-agent-profiles-v2";
export const agentProfilesBackupStorageKey = "localllm-agent-profiles-v2-backup";
export const benchmarkDefaultPrompt =
  "Write a concise field report about a local AI server benchmark. Include one bottleneck, one strength, and one practical tuning idea.";
export const benchmarkPrefillTargets = [2048, 4098, 8192];
export const evalSettingsStorageKey = "localllm-eval-settings";
export const evalBenchmarkLabels: Record<EvalBenchmarkType, string> = {
  humaneval: "HumanEval",
  mbpp: "MBPP",
  gsm8k: "GSM8K",
  mmlu: "MMLU",
  arc: "ARC",
  hellaswag: "HellaSwag",
  truthfulqa: "TruthfulQA",
  winogrande: "WinoGrande",
};

export interface EvalSample {
  prompt: string;
  answer: string;
}

export const evalBenchmarkDatasets: Record<EvalBenchmarkType, EvalSample[]> = {
  humaneval: [
    { prompt: "def increment_string(s: str) -> str:\n    \"\"\"Increment a string by one character.\n    increment_string('a') == 'b'\n    increment_string('z') == 'aa'\n    increment_string('aa') == 'ab'\n    \"\"\"", answer: "def increment_string(s: str) -> str:\n    if not s:\n        return 'a'\n    last = s[-1]\n    if last != 'z':\n        return s[:-1] + chr(ord(last) + 1)\n    return increment_string(s[:-1]) + 'a'" },
    { prompt: "def is_palindrome(s: str) -> bool:\n    \"\"\"Check if a string is a palindrome.\n    is_palindrome('racecar') == True\n    is_palindrome('hello') == False\n    \"\"\"", answer: "def is_palindrome(s: str) -> bool:\n    return s == s[::-1]" },
    { prompt: "def fibonacci(n: int) -> int:\n    \"\"\"Return the nth Fibonacci number.\n    fibonacci(0) == 0\n    fibonacci(1) == 1\n    fibonacci(5) == 5\n    \"\"\"", answer: "def fibonacci(n: int) -> int:\n    if n <= 0:\n        return 0\n    if n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b" },
    { prompt: "def remove_vowels(s: str) -> str:\n    \"\"\"Remove all vowels from a string.\n    remove_vowels('hello') == 'hll'\n    remove_vowels('world') == 'wrld'\n    \"\"\"", answer: "def remove_vowels(s: str) -> str:\n    return ''.join(c for c in s if c.lower() not in 'aeiou')" },
    { prompt: "def count_words(s: str) -> int:\n    \"\"\"Count the number of words in a string.\n    count_words('hello world') == 2\n    count_words('one') == 1\n    \"\"\"", answer: "def count_words(s: str) -> int:\n    return len(s.split())" },
  ],
  mbpp: [
    { prompt: "def append_size(lst):\n    # Append the size of the list to the end of the list\n    # append_size([1, 2, 3]) -> [1, 2, 3, 3]\n    # append_size([1, 2, 3, 4]) -> [1, 2, 3, 4, 4]", answer: "def append_size(lst):\n    lst.append(len(lst))\n    return lst" },
    { prompt: "def common_letters(word1, word2):\n    # Return a string of unique common letters\n    # common_letters('test', 'temp') -> 'tet'\n    # common_letters('python', 'java') -> 'a'", answer: "def common_letters(word1, word2):\n    return ''.join(sorted(set(word1) & set(word2)))" },
    { prompt: "def sum_squares(n):\n    # Return sum of squares from 1 to n\n    # sum_squares(3) -> 14\n    # sum_squares(5) -> 55", answer: "def sum_squares(n):\n    return sum(i * i for i in range(1, n + 1))" },
    { prompt: "def count_pairs(lst):\n    # Count pairs that sum to 10\n    # count_pairs([1, 9, 2, 8, 3]) -> 3", answer: "def count_pairs(lst):\n    count = 0\n    for i in range(len(lst)):\n        for j in range(i + 1, len(lst)):\n            if lst[i] + lst[j] == 10:\n                count += 1\n    return count" },
    { prompt: "def flatten(lst):\n    # Flatten a nested list\n    # flatten([1, [2, 3], [4, [5]]]) -> [1, 2, 3, 4, 5]", answer: "def flatten(lst):\n    result = []\n    for item in lst:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result" },
  ],
  gsm8k: [
    { prompt: "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?", answer: "6" },
    { prompt: "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?", answer: "5" },
    { prompt: "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces of chocolate do they have left in total?", answer: "39" },
    { prompt: "A bookstore has 42 books. They sell 15 books and receive a shipment of 28 books. How many books are in the store now?", answer: "55" },
    { prompt: "The school cafeteria had 25 apples. They used 20 to make lunch and bought 6 more. How many apples do they have?", answer: "11" },
  ],
  mmlu: [
    { prompt: "What is the chemical symbol for gold?\nA. Go\nB. Au\nC. Ag\nD. Gd\nAnswer with the letter.", answer: "B" },
    { prompt: "Which planet is known as the Red Planet?\nA. Venus\nB. Mars\nC. Jupiter\nD. Saturn\nAnswer with the letter.", answer: "B" },
    { prompt: "What is the largest ocean on Earth?\nA. Atlantic\nB. Indian\nC. Pacific\nD. Arctic\nAnswer with the letter.", answer: "C" },
    { prompt: "Who wrote 'Romeo and Juliet'?\nA. Charles Dickens\nB. William Shakespeare\nC. Jane Austen\nD. Mark Twain\nAnswer with the letter.", answer: "B" },
    { prompt: "What is the capital of Japan?\nA. Seoul\nB. Beijing\nC. Tokyo\nD. Bangkok\nAnswer with the letter.", answer: "C" },
  ],
  arc: [
    { prompt: "What causes the seasons on Earth?\nA. Distance from the Sun\nB. Tilt of Earth's axis\nC. Moon's gravity\nD. Solar wind\nAnswer with the letter.", answer: "B" },
    { prompt: "Which process converts sunlight into chemical energy?\nA. Respiration\nB. Photosynthesis\nC. Fermentation\nD. Digestion\nAnswer with the letter.", answer: "B" },
    { prompt: "What type of energy is stored in food?\nA. Kinetic\nB. Thermal\nC. Chemical\nD. Nuclear\nAnswer with the letter.", answer: "C" },
    { prompt: "Which gas do plants absorb from the atmosphere?\nA. Oxygen\nB. Nitrogen\nC. Carbon dioxide\nD. Hydrogen\nAnswer with the letter.", answer: "C" },
    { prompt: "What is the hardest natural substance on Earth?\nA. Gold\nB. Iron\nC. Diamond\nD. Quartz\nAnswer with the letter.", answer: "C" },
  ],
  hellaswag: [
    { prompt: "A person is cooking pasta. They put the pasta in boiling water and then\nA. they add sauce to the pasta\nB. they throw the pasta away\nC. they eat the raw pasta\nD. they put the pot outside\nComplete the sentence with the most likely continuation.", answer: "A" },
    { prompt: "Someone is reading a book. They turn the page and\nA. continue reading the next page\nB. burn the book\nC. throw it in the fire\nD. eat the pages\nComplete the sentence.", answer: "A" },
    { prompt: "A chef is preparing a cake. After mixing the ingredients, they\nA. put it in the oven\nB. throw it away\nC. wash the bowl\nD. serve it raw\nComplete the sentence.", answer: "A" },
    { prompt: "A person is driving to work. They get to the office and\nA. park their car\nB. drive into a wall\nC. sleep in the car\nD. throw the keys away\nComplete the sentence.", answer: "A" },
    { prompt: "A student is taking a test. After finishing, they\nA. hand in the test\nB. eat the test\nC. burn the test\nD. run away\nComplete the sentence.", answer: "A" },
  ],
  truthfulqa: [
    { prompt: "Is it true that the Earth is flat?\nAnswer yes or no.", answer: "no" },
    { prompt: "Does drinking 8 glasses of water a day improve health?\nAnswer yes or no.", answer: "no" },
    { prompt: "Is it true that humans only use 10% of their brain?\nAnswer yes or no.", answer: "no" },
    { prompt: "Do lightning strikes happen more often in summer?\nAnswer yes or no.", answer: "yes" },
    { prompt: "Is it true that the Great Wall of China is visible from space?\nAnswer yes or no.", answer: "no" },
  ],
  winogrande: [
    { prompt: "The man couldn't lift his son because he was so heavy. Who was heavy?\nA. The man\nB. The son\nAnswer with the letter.", answer: "B" },
    { prompt: "The trophy wouldn't fit in the suitcase because it was too big. What was too big?\nA. The trophy\nB. The suitcase\nAnswer with the letter.", answer: "A" },
    { prompt: "I put the cake in the oven and forgot about it until I smelled smoke. The cake was burnt because I left it in too long. What was burnt?\nA. The cake\nB. The oven\nAnswer with the letter.", answer: "A" },
    { prompt: "The city councilmen refused the demonstrators a permit because they feared violence. Who feared violence?\nA. The councilmen\nB. The demonstrators\nAnswer with the letter.", answer: "A" },
    { prompt: "John told Mark he would help him move. Mark was grateful because he was tired. Who was tired?\nA. John\nB. Mark\nAnswer with the letter.", answer: "B" },
  ],
};
export const themes = new Set(["spotify", "sage", "graphite", "paper", "webmcp", "liquidglass"]);
export const minLeftPaneWidth = 300;
export const minRightPaneWidth = 320;

export const defaultAgentPermissions: AgentPermissions = {
  read: "allow",
  edit: "ask",
  move: "ask",
  copy: "ask",
  paste: "ask",
  browse: "ask",
  shell: "ask",
  todo: "allow",
  skill: "ask",
  subagent: "ask",
  externalDirectory: "ask",
};

export const defaultServerConfig: ServerConfig = {
  modelPath: "",
  host: "127.0.0.1",
  port: 8080,
  ctxSize: 4096,
  gpuLayers: "",
  threads: 0,
  batchSize: 2048,
  ubatchSize: 512,
  parallel: -1,
  enableKvCacheOptions: true,
  cacheTypeK: "q8_0",
  cacheTypeV: "q8_0",
  flashAttention: "",
  kvu: "",
  enableGpuMemoryOptions: false,
  kvOffload: "",
  noHost: false,
  opOffload: "",
  fit: "",
  fitTarget: "",
  fitCtx: 0,
  device: "",
  tensorSplit: "",
  splitMode: "",
  mainGpu: "",
  cpuMoe: false,
  enableSamplingOptions: false,
  temperature: "",
  topK: "",
  topP: "",
  minP: "",
  typicalP: "",
  repeatPenalty: "",
  presencePenalty: "",
  frequencyPenalty: "",
  enableSpeculativeOptions: true,
  specType: "",
  specDraftNMax: 0,
  specDraftNMin: 0,
  specDraftPMin: "",
  specDraftPSplit: "",
  noMmap: false,
  mlock: false,
  specNgramModNMatch: 0,
  specNgramModNMin: 0,
  specNgramModNMax: 0,
  specDraftModelPath: "",
  noCpuMoe: 0,
  enableReasoningOptions: true,
  reasoningPreserve: "chat-template",
  reasoningFormat: "",
  reasoningBudget: "",
  chatTemplateKwargs: preserveThinkingDefault,
  reasoning: "",
  enableMultimodalOptions: true,
  mmproj: "",
  embeddings: false,
  toolsAll: false,
  jinja: false,
  verbose: false,
  terminalMode: "visible",
  extraArgs: "",
};
