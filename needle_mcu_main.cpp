#include <TensorFlowLite_ESP32.h> // 或 #include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "needle_model_data.h" // 稍早 xxd 導出的模型陣列

// 1. 定義靜態 Tensor Arena (SRAM 靜態記憶體池，MCU 不使用 malloc)
constexpr int kTensorArenaSize = 64 * 1024; // 64 KB
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output_logits = nullptr;

void setup() {
    Serial.begin(115200);
    Serial.println("🌲 Needle 3 TFLM Microcontroller Booting...");
    // 2. 從 Flash 載入 FlatBuffers 模型
    model = tflite::GetModel(g_needle_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("❌ 模型版本不匹配！");
        return;
    }
    // 3. 註冊 Micro 算子解析器
    static tflite::AllOpsResolver resolver;
    // 4. 建立 MicroInterpreter (零 malloc 分配)
    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;
    // 5. 分配張量記憶體 (由 Greedy Memory Planner 在 tensor_arena 中規劃)
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("❌ AllocateTensors 失敗！請增大 kTensorArenaSize");
        return;
    }
    input = interpreter->input(0);
    output_logits = interpreter->output(0);
    Serial.println("✅ TFLM 推論引擎初始化成功，已進入即時監聽狀態！");
}

void loop() {
    // 6. 觸發推論 (Invoke)
    if (interpreter->Invoke() == kTfLiteOk) {
        Serial.println("⚡ 推論成功，執行對應 GPIO / PWM 控制！");
    }
    delay(2000);
}