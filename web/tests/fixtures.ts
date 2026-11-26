export const PNG_DATA_URL_STUB = 'data:image/png;base64,iVBORw0KGgo=';

export const modelsJson = [
  {
    id: 'pneumoniamnist-convnext',
    scan_type: 'Pediatric chest X-ray',
    modality: 'X-ray',
    question: 'Pneumonia or normal?',
    label_mode: 'binary',
    classes: ['normal', 'pneumonia'],
    input: { channels: 1, size: 224 },
    dataset: 'PneumoniaMNIST (MedMNIST v2)',
    license: 'CC BY 4.0',
    citation:
      'Yang, J. et al. MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification. Scientific Data, 2023.',
    run_id: '20261127-031500-abcd1234',
    test_auroc: { value: 0.97, lower: 0.96, upper: 0.98 },
    demo_count: 4,
  },
  {
    id: 'dermamnist-resnet',
    scan_type: 'Dermatoscope image',
    modality: 'Dermatoscopy',
    question: 'Which of 7 skin lesion types?',
    label_mode: 'multiclass',
    classes: [
      'actinic_keratosis',
      'basal_cell_carcinoma',
      'benign_keratosis',
      'dermatofibroma',
      'melanoma',
      'nevus',
      'vascular_lesion',
    ],
    input: { channels: 3, size: 224 },
    dataset: 'DermaMNIST (MedMNIST v2)',
    license: 'CC BY-NC 4.0',
    citation:
      'Yang, J. et al. MedMNIST v2. Scientific Data, 2023. Tschandl, P. et al. The HAM10000 dataset. Scientific Data, 2018.',
    run_id: '20261127-041500-ef567890',
    test_auroc: null,
    demo_count: 0,
  },
];

export function demosJson(modelId: string) {
  if (modelId !== 'pneumoniamnist-convnext') return [];
  return ['test-12', 'test-40', 'test-77', 'test-103'].map((sampleId, i) => ({
    sample_id: sampleId,
    truth: i % 2 === 0 ? 'pneumonia' : 'normal',
    image_url: `/models/${modelId}/demo/${sampleId}.png`,
  }));
}

export function predictionJson({
  truth,
  inputPng,
  attributionPng,
}: {
  truth: string | null;
  inputPng: string;
  attributionPng: string;
}) {
  return {
    model: 'pneumoniamnist-convnext',
    label_mode: 'binary',
    probabilities: [
      { class: 'normal', probability: 0.08 },
      { class: 'pneumonia', probability: 0.92 },
    ],
    predicted: ['pneumonia'],
    threshold: 0.41,
    input_png: inputPng,
    attribution_png: attributionPng,
    truth,
    disclaimer: 'Research model, not a medical device. Not for diagnosis or treatment.',
    latency_ms: 840,
  };
}
