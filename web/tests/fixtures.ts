export const PNG_DATA_URL_STUB = 'data:image/png;base64,iVBORw0KGgo=';

export const modelsJson = [
  {
    id: 'pneumoniamnist-convnext',
    scan_type: 'Pediatric chest X-ray',
    modality: 'X-ray',
    category: 'X-ray and mammography',
    question: 'Pneumonia or normal?',
    label_mode: 'binary',
    classes: ['normal', 'pneumonia'],
    input: { channels: 1, size: 224, dims: 2 },
    dataset: 'PneumoniaMNIST (MedMNIST v2)',
    license: 'CC BY 4.0',
    citation:
      'Yang, J. et al. MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification. Scientific Data, 2023.',
    run_id: '20261127-031500-abcd1234',
    test_auroc: { value: 0.97, lower: 0.96, upper: 0.98 },
    demo_count: 4,
    weights: {
      url: 'https://diagnostics.harville.dev/opendiagnostics-weights/pneumoniamnist-convnext/20261127-031500-abcd1234/checkpoint.ckpt',
      bytes: 200_123_456,
      sha256: 'a'.repeat(64),
    },
  },
  {
    id: 'dermamnist-resnet',
    scan_type: 'Dermatoscope image',
    modality: 'Dermatoscopy',
    category: 'Skin',
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
    input: { channels: 3, size: 224, dims: 2 },
    dataset: 'DermaMNIST (MedMNIST v2)',
    license: 'CC BY-NC 4.0',
    citation:
      'Yang, J. et al. MedMNIST v2. Scientific Data, 2023. Tschandl, P. et al. The HAM10000 dataset. Scientific Data, 2018.',
    run_id: '20261127-041500-ef567890',
    test_auroc: null,
    demo_count: 0,
    weights: null,
  },
  {
    id: 'synapsemnist3d-resnet18',
    scan_type: 'Synapse electron microscopy volume',
    modality: 'Electron microscopy',
    category: 'Pathology and microscopy',
    question: 'Excitatory or inhibitory synapse?',
    label_mode: 'binary',
    classes: ['inhibitory synapse', 'excitatory synapse'],
    input: { channels: 1, size: 64, dims: 3 },
    dataset: 'synapsemnist3d (MedMNIST v2)',
    license: 'CC BY 4.0',
    citation: 'Yang, J. et al. MedMNIST v2. Scientific Data, 2023.',
    run_id: '20261127-090000-12ea1f46',
    test_auroc: { value: 0.985, lower: 0.97, upper: 0.995 },
    demo_count: 2,
    weights: null,
  },
];

export function demosJson(modelId: string) {
  if (modelId === 'synapsemnist3d-resnet18') {
    return ['test-3', 'test-9'].map((sampleId, i) => ({
      sample_id: sampleId,
      truth: i === 0 ? 'excitatory synapse' : 'inhibitory synapse',
      image_url: `/models/${modelId}/demo/${sampleId}.png`,
    }));
  }
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
  slices = null,
}: {
  truth: string | null;
  inputPng: string;
  attributionPng: string;
  slices?: { input: string[]; attribution: string[] } | null;
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
    slices,
    truth,
    disclaimer: 'Research model, not a medical device. Not for diagnosis or treatment.',
    latency_ms: 840,
  };
}
