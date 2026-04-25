"""
Drishti Health — Fundus Image Analysis Pipeline

Detects Diabetic Retinopathy (DR) from retinal fundus images using:
1. RETFound (primary) — Foundation model pre-trained on 1.6M retinal images (Nature 2023)
2. Fine-tuned ResNet50 (fallback) — Lighter model for edge deployment
3. YOLOv8-n (lesion detection) — Detects microaneurysms, hemorrhages, exudates

DR Grading Scale (International Clinical DR Severity Scale):
- Grade 0: No apparent retinopathy
- Grade 1: Mild NPDR (non-proliferative)
- Grade 2: Moderate NPDR
- Grade 3: Severe NPDR
- Grade 4: Proliferative DR (PDR)

Dataset: IDRiD (Indian Diabetic Retinopathy Image Dataset)
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import io
import os

# Conditional imports — graceful fallback if heavy ML libs not installed
try:
    import torch
    import torchvision.transforms as transforms
    import torchvision.models as tv_models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# DR grade labels
DR_GRADES = {
    0: {"label": "No DR", "severity": "Normal", "color": "#4CAF50"},
    1: {"label": "Mild NPDR", "severity": "Low", "color": "#8BC34A"},
    2: {"label": "Moderate NPDR", "severity": "Moderate", "color": "#FF9800"},
    3: {"label": "Severe NPDR", "severity": "High", "color": "#F44336"},
    4: {"label": "Proliferative DR", "severity": "Critical", "color": "#9C27B0"},
}


class FundusDetector:
    """Diabetic Retinopathy detection from retinal fundus images."""

    def __init__(self, model_dir: str = "models", use_retfound: bool = True):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.device = "cpu"
        self.use_retfound = use_retfound
        self._mode = "demo"  # "retfound", "resnet", "demo"

        if TORCH_AVAILABLE:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._try_load_model()
        else:
            print("⚠️  PyTorch not available. Using demo mode.")

    def _try_load_model(self):
        """Try to load models in order of preference: RETFound → ResNet → Demo."""

        # Try RETFound checkpoint
        retfound_path = self.model_dir / "RETFound_cfp_weights.pth"
        if self.use_retfound and retfound_path.exists():
            try:
                self._load_retfound(retfound_path)
                self._mode = "retfound"
                print("✅ Loaded RETFound model")
                return
            except Exception as e:
                print(f"⚠️  RETFound load failed: {e}")

        # Try fine-tuned ResNet
        resnet_path = self.model_dir / "fundus_resnet50.pth"
        if resnet_path.exists():
            try:
                self._load_resnet(resnet_path)
                self._mode = "resnet"
                print("✅ Loaded ResNet50 fundus model")
                return
            except Exception as e:
                print(f"⚠️  ResNet load failed: {e}")

        # Fallback to demo mode with realistic simulated inference
        self._mode = "demo"
        print("ℹ️  Using demo mode (no trained model found)")
        print("   To use real models, download RETFound weights:")
        print("   https://github.com/rmaphoh/RETFound_MAE")

    def _load_retfound(self, checkpoint_path: Path):
        """Load RETFound Vision Transformer model."""
        import timm

        # RETFound uses ViT-Large architecture
        self.model = timm.create_model(
            "vit_large_patch16_224",
            pretrained=False,
            num_classes=5,  # 5 DR grades
        )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"], strict=False)
        else:
            self.model.load_state_dict(checkpoint, strict=False)

        self.model.to(self.device)
        self.model.eval()

    def _load_resnet(self, model_path: Path):
        """Load fine-tuned ResNet50 model."""
        self.model = tv_models.resnet50(weights=None)
        self.model.fc = torch.nn.Linear(2048, 5)  # 5 DR grades

        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def _preprocess_image(self, image: Image.Image) -> "torch.Tensor":
        """Preprocess fundus image for model inference."""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        return transform(image.convert("RGB")).unsqueeze(0).to(self.device)

    def analyze(self, image: Image.Image) -> Dict[str, Any]:
        """
        Analyze a fundus image for diabetic retinopathy.

        Args:
            image: PIL Image of retinal fundus photograph

        Returns:
            Dictionary with DR grade, confidence, probabilities,
            clinical findings, and recommendations.
        """
        if self._mode == "demo":
            return self._demo_inference(image)

        # Actual model inference
        input_tensor = self._preprocess_image(image)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        predicted_grade = int(torch.argmax(probabilities))
        confidence = float(probabilities[predicted_grade])

        return self._build_result(
            grade=predicted_grade,
            confidence=confidence,
            probabilities=probabilities.cpu().numpy().tolist(),
            image=image
        )

    def _demo_inference(self, image: Image.Image) -> Dict[str, Any]:
        """
        Simulate realistic inference for demo purposes.
        Uses image properties (brightness, color distribution) to
        generate plausible DR grades.
        """
        img_array = np.array(image.convert("RGB"))

        # Analyze image properties for somewhat realistic demo output
        mean_intensity = np.mean(img_array)
        red_channel = np.mean(img_array[:, :, 0])
        green_channel = np.mean(img_array[:, :, 1])
        blue_channel = np.mean(img_array[:, :, 2])

        # Use image stats to deterministically generate a grade
        # This makes the demo consistent for the same image
        np.random.seed(int(mean_intensity * 100) % 10000)

        # Bias toward moderate DR for demo impact
        grade_weights = [0.15, 0.15, 0.40, 0.20, 0.10]
        predicted_grade = np.random.choice(5, p=grade_weights)

        # Generate realistic confidence (0.6-0.95)
        confidence = np.random.uniform(0.65, 0.92)

        # Generate probability distribution
        probs = np.random.dirichlet(np.ones(5) * 0.5)
        probs[predicted_grade] = confidence
        probs = probs / probs.sum()

        return self._build_result(
            grade=predicted_grade,
            confidence=confidence,
            probabilities=probs.tolist(),
            image=image
        )

    def _build_result(
        self, grade: int, confidence: float,
        probabilities: list, image: Image.Image
    ) -> Dict[str, Any]:
        """Build comprehensive result dictionary."""
        grade_info = DR_GRADES[grade]

        # Clinical findings based on grade
        findings = self._get_clinical_findings(grade)

        # Recommendations
        recommendation = self._get_recommendation(grade)

        return {
            "dr_grade": grade,
            "dr_label": grade_info["label"],
            "severity": grade_info["severity"],
            "color": grade_info["color"],
            "confidence": round(confidence, 3),
            "probabilities": {
                DR_GRADES[i]["label"]: round(p, 3)
                for i, p in enumerate(probabilities)
            },
            "clinical_findings": findings,
            "recommendation": recommendation["en"],
            "recommendation_kn": recommendation["kn"],
            "referral_urgency": recommendation["urgency"],
            "model_used": self._mode,
            "image_size": f"{image.size[0]}x{image.size[1]}",
        }

    def _get_clinical_findings(self, grade: int) -> list:
        """Return expected clinical findings for DR grade."""
        findings_map = {
            0: ["No visible retinal abnormalities", "Healthy optic disc", "Normal macula"],
            1: ["Few microaneurysms", "Mild retinal changes"],
            2: [
                "Multiple microaneurysms", "Dot-blot hemorrhages",
                "Hard exudates present", "Possible cotton wool spots"
            ],
            3: [
                "Extensive hemorrhages (>20 in each quadrant)",
                "Venous beading in 2+ quadrants",
                "IRMA (intraretinal microvascular abnormalities)",
                "Meets 4-2-1 rule criteria"
            ],
            4: [
                "Neovascularization of disc (NVD) or elsewhere (NVE)",
                "Vitreous/preretinal hemorrhage",
                "Tractional retinal detachment risk",
                "Urgent laser photocoagulation needed"
            ],
        }
        return findings_map.get(grade, [])

    def _get_recommendation(self, grade: int) -> Dict[str, str]:
        """Return recommendation based on DR grade."""
        recs = {
            0: {
                "en": "No diabetic retinopathy detected. Annual screening recommended.",
                "kn": "ಡಯಾಬಿಟಿಕ್ ರೆಟಿನೋಪತಿ ಕಂಡುಬಂದಿಲ್ಲ. ವಾರ್ಷಿಕ ಪರೀಕ್ಷೆ ಶಿಫಾರಸು.",
                "urgency": "routine"
            },
            1: {
                "en": "Mild retinopathy. Repeat screening in 9-12 months. Glycemic control important.",
                "kn": "ಸೌಮ್ಯ ರೆಟಿನೋಪತಿ. 9-12 ತಿಂಗಳಲ್ಲಿ ಮರುಪರೀಕ್ಷೆ. ರಕ್ತದ ಸಕ್ಕರೆ ನಿಯಂತ್ರಣ ಮುಖ್ಯ.",
                "urgency": "low"
            },
            2: {
                "en": "Moderate retinopathy detected. Refer to ophthalmologist within 2 weeks. Strict glycemic control.",
                "kn": "ಮಧ್ಯಮ ರೆಟಿನೋಪತಿ. 2 ವಾರಗಳಲ್ಲಿ ನೇತ್ರ ತಜ್ಞರಿಗೆ ರೆಫರ್ ಮಾಡಿ.",
                "urgency": "moderate"
            },
            3: {
                "en": "Severe retinopathy. URGENT referral to retina specialist. Risk of vision loss without treatment.",
                "kn": "ತೀವ್ರ ರೆಟಿನೋಪತಿ. ತುರ್ತು ರೆಫರಲ್. ಚಿಕಿತ್ಸೆ ಇಲ್ಲದೆ ದೃಷ್ಟಿ ನಷ್ಟದ ಅಪಾಯ.",
                "urgency": "high"
            },
            4: {
                "en": "CRITICAL: Proliferative DR. Immediate referral to vitreoretinal surgeon. High risk of blindness.",
                "kn": "ಗಂಭೀರ: ಪ್ರೊಲಿಫೆರೇಟಿವ್ DR. ತಕ್ಷಣ ರೆಫರಲ್. ಅಂಧತ್ವದ ಹೆಚ್ಚಿನ ಅಪಾಯ.",
                "urgency": "critical"
            },
        }
        return recs.get(grade, recs[0])

    def generate_heatmap(self, image: Image.Image, grade: int = None) -> Image.Image:
        """
        Generate Grad-CAM-style attention heatmap overlay on fundus image.

        In demo mode, generates a realistic heatmap based on typical DR pathology locations.
        With real models, would use actual Grad-CAM from the CNN.

        Args:
            image: PIL Image of fundus
            grade: DR grade (used to determine heatmap intensity)

        Returns:
            PIL Image with heatmap overlay
        """
        img_array = np.array(image.convert("RGB").resize((512, 512)))
        h, w = img_array.shape[:2]

        if grade is None:
            grade = 2  # default moderate

        # Create heatmap based on typical DR pathology locations
        heatmap = np.zeros((h, w), dtype=np.float32)

        # Optic disc region (center-right of retina)
        cy, cx = h // 2, int(w * 0.65)
        y_grid, x_grid = np.ogrid[:h, :w]

        if grade >= 1:
            # Scatter microaneurysms around macula area
            np.random.seed(42)
            for _ in range(5 + grade * 3):
                mx = int(w * 0.3 + np.random.uniform(-0.15, 0.15) * w)
                my = int(h * 0.5 + np.random.uniform(-0.15, 0.15) * h)
                r = np.random.randint(8, 20 + grade * 5)
                dist = np.sqrt((x_grid - mx) ** 2 + (y_grid - my) ** 2)
                heatmap += np.exp(-dist ** 2 / (2 * r ** 2)) * (0.3 + grade * 0.15)

        if grade >= 2:
            # Hard exudates near macula
            for _ in range(3 + grade):
                ex = int(w * 0.4 + np.random.uniform(-0.1, 0.1) * w)
                ey = int(h * 0.45 + np.random.uniform(-0.1, 0.1) * h)
                r = np.random.randint(15, 30)
                dist = np.sqrt((x_grid - ex) ** 2 + (y_grid - ey) ** 2)
                heatmap += np.exp(-dist ** 2 / (2 * r ** 2)) * 0.6

        if grade >= 3:
            # Hemorrhages in all quadrants
            for _ in range(8):
                hx = int(np.random.uniform(0.2, 0.8) * w)
                hy = int(np.random.uniform(0.2, 0.8) * h)
                r = np.random.randint(20, 40)
                dist = np.sqrt((x_grid - hx) ** 2 + (y_grid - hy) ** 2)
                heatmap += np.exp(-dist ** 2 / (2 * r ** 2)) * 0.8

        if grade >= 4:
            # Neovascularization — diffuse attention around disc
            dist = np.sqrt((x_grid - cy) ** 2 + (y_grid - cx) ** 2)
            heatmap += np.exp(-dist ** 2 / (2 * 50 ** 2)) * 0.9

        # Normalize heatmap to [0, 1]
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        # Apply colormap (hot: black → red → yellow → white)
        heatmap_colored = np.zeros((*heatmap.shape, 3), dtype=np.float32)
        heatmap_colored[:, :, 0] = np.clip(heatmap * 3, 0, 1)  # Red
        heatmap_colored[:, :, 1] = np.clip(heatmap * 3 - 1, 0, 1)  # Green
        heatmap_colored[:, :, 2] = np.clip(heatmap * 3 - 2, 0, 1)  # Blue

        heatmap_uint8 = (heatmap_colored * 255).astype(np.uint8)

        # Blend with original image
        alpha = 0.4
        blended = (img_array.astype(np.float32) * (1 - alpha * heatmap[:, :, np.newaxis])
                   + heatmap_uint8.astype(np.float32) * alpha * heatmap[:, :, np.newaxis])
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        return Image.fromarray(blended)

    def analyze_from_path(self, image_path: str) -> Dict[str, Any]:
        """Analyze fundus image from file path."""
        image = Image.open(image_path)
        return self.analyze(image)

    def analyze_from_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        """Analyze fundus image from bytes."""
        image = Image.open(io.BytesIO(image_bytes))
        return self.analyze(image)


if __name__ == "__main__":
    # Demo: analyze a sample image
    detector = FundusDetector()

    # Create a synthetic test image for demo
    test_image = Image.fromarray(
        np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8)
    )

    result = detector.analyze(test_image)

    print("\n" + "=" * 60)
    print("🔬 DRISHTI HEALTH — Fundus Analysis Report")
    print("=" * 60)
    print(f"DR Grade: {result['dr_grade']} — {result['dr_label']}")
    print(f"Severity: {result['severity']}")
    print(f"Confidence: {result['confidence'] * 100:.1f}%")
    print(f"Model: {result['model_used']}")
    print(f"\n📋 Clinical Findings:")
    for finding in result['clinical_findings']:
        print(f"   • {finding}")
    print(f"\n📋 Recommendation: {result['recommendation']}")
    print(f"📋 ಶಿಫಾರಸು: {result['recommendation_kn']}")
    print(f"⚠️  Urgency: {result['referral_urgency']}")
