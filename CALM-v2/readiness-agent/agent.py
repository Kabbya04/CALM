# File: agent.py (CALM-v2 Enhanced Version)
# ReadinessAgent adapted for CALM-v2 with ensemble calibration support

import numpy as np
from sklearn.metrics import confusion_matrix
import warnings

class ReadinessAgent:
    """
    Enhanced ReadinessAgent for CALM-v2 framework.
    Works with ensemble confidence calibration and hierarchical memory.
    """
    def __init__(self, threshold=0.7, adaptive=True, target_deployment_rate=0.7):
        """
        Enhanced ReadinessAgent with adaptive thresholding.
        
        Args:
            threshold (float): Initial confidence threshold (0-1)
            adaptive (bool): Whether to adapt threshold based on validation data
            target_deployment_rate (float): Target percentage of predictions to deploy (0-1)
        """
        if not 0 < threshold < 1:
            raise ValueError("Threshold must be between 0 and 1.")
        
        self.threshold = threshold
        self.adaptive = adaptive
        self.target_deployment_rate = target_deployment_rate
        self.calibrated_threshold = threshold
        print(f"ReadinessAgent initialized:")
        print(f"  Initial threshold: {self.threshold:.2%}")
        print(f"  Adaptive: {self.adaptive}")
        print(f"  Target deployment rate: {self.target_deployment_rate:.2%}")

    def calibrate_threshold(self, confidences, predictions, true_labels, verbose=True):
        """
        Calibrate the threshold based on validation data to achieve target deployment rate
        while maintaining good safety metrics.
        
        Args:
            confidences: Array of confidence scores (can be from ensemble)
            predictions: Array of model predictions  
            true_labels: Array of true labels
            verbose: Whether to print calibration details
        """
        confidences = np.array(confidences)
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        
        if not self.adaptive:
            self.calibrated_threshold = self.threshold
            return
        
        if verbose:
            print(f"\n=== Adaptive Threshold Calibration ===")
            print(f"Confidence range: [{np.min(confidences):.3f}, {np.max(confidences):.3f}]")
            print(f"Confidence mean: {np.mean(confidences):.3f} ± {np.std(confidences):.3f}")
        
        # Try different thresholds and find the best one
        candidate_thresholds = np.percentile(confidences, np.linspace(10, 95, 18))
        best_threshold = self.threshold
        best_score = -np.inf
        
        results = []
        
        for thresh in candidate_thresholds:
            # Simulate decisions with this threshold
            decisions = ["Deploy" if conf >= thresh else "Request Feedback" for conf in confidences]
            metrics = self.get_metrics(predictions, true_labels, decisions)
            
            # Calculate deployment rate
            deployment_rate = metrics['Total Deployed']
            
            # Skip if deployment rate is too extreme
            if deployment_rate < 0.1 or deployment_rate > 0.95:
                continue
            
            # Composite score balancing multiple objectives
            # Prefer thresholds that:
            # 1. Achieve close to target deployment rate
            # 2. Have low False Positive Rate (safety)
            # 3. Have high Deployment Decision Accuracy
            deployment_score = 1 - abs(deployment_rate - self.target_deployment_rate)
            safety_score = 1 - metrics['False Positive Rate (FPR)']
            accuracy_score = metrics['Deployment Decision Accuracy']
            
            # Weighted combination (safety is most important)
            composite_score = 0.3 * deployment_score + 0.4 * safety_score + 0.3 * accuracy_score
            
            results.append({
                'threshold': thresh,
                'score': composite_score,
                'deployment_rate': deployment_rate,
                'fpr': metrics['False Positive Rate (FPR)'],
                'accuracy': metrics['Deployment Decision Accuracy']
            })
            
            if composite_score > best_score:
                best_score = composite_score
                best_threshold = thresh
        
        self.calibrated_threshold = best_threshold
        
        if verbose:
            print(f"\nThreshold calibration results:")
            print(f"  Original threshold: {self.threshold:.3f}")
            print(f"  Calibrated threshold: {self.calibrated_threshold:.3f}")
            
            # Show performance at calibrated threshold
            decisions = ["Deploy" if conf >= self.calibrated_threshold else "Request Feedback" for conf in confidences]
            final_metrics = self.get_metrics(predictions, true_labels, decisions)
            
            print(f"  Expected deployment rate: {final_metrics['Total Deployed']:.2%}")
            print(f"  Expected FPR: {final_metrics['False Positive Rate (FPR)']:.2%}")
            print(f"  Expected decision accuracy: {final_metrics['Deployment Decision Accuracy']:.2%}")
            
            # Warning if no good threshold found
            if best_score < 0.5:
                print(f"  WARNING: Low calibration score ({best_score:.3f}). Consider:")
                print(f"    - Improving base model accuracy")
                print(f"    - Using better confidence calibration")
                print(f"    - Adjusting target deployment rate")

    def decide(self, confidence, agreement=None):
        """
        Makes a decision based on the calibrated confidence threshold.
        Optionally considers ensemble agreement for CALM-v2.
        
        Args:
            confidence: Confidence score (from ensemble)
            agreement: Optional ensemble agreement score (for CALM-v2 tiered decisions)
        
        Returns:
            "Deploy" or "Request Feedback"
        """
        # If agreement is provided, use tiered decision logic
        if agreement is not None:
            # High confidence with high agreement -> deploy
            if confidence >= 0.85 and agreement > 0.7:
                return "Deploy"
            # Medium confidence with high agreement -> deploy
            elif confidence >= 0.55 and agreement > 0.8:
                return "Deploy"
            # Otherwise request feedback
            else:
                return "Request Feedback"
        
        # Standard threshold-based decision
        if confidence >= self.calibrated_threshold:
            return "Deploy"
        else:
            return "Request Feedback"

    @staticmethod
    def get_metrics(model_predictions, true_labels, agent_decisions):
        """
        Enhanced metrics calculation with additional insights.
        """
        model_predictions = np.array(model_predictions)
        true_labels = np.array(true_labels)
        agent_decisions = np.array(agent_decisions)
        
        model_correct = (model_predictions == true_labels)
        
        # Basic metrics
        correct_deploys = np.sum((agent_decisions == "Deploy") & model_correct)
        correct_requests = np.sum((agent_decisions == "Request Feedback") & ~model_correct)
        decision_accuracy = (correct_deploys + correct_requests) / len(true_labels)
        
        # False Positive Rate
        total_model_errors = np.sum(~model_correct)
        incorrect_deploys = np.sum((agent_decisions == "Deploy") & ~model_correct)
        fpr = incorrect_deploys / total_model_errors if total_model_errors > 0 else 0.0
        
        # Additional metrics
        total_deployed = np.mean(agent_decisions == "Deploy")
        total_requested = np.mean(agent_decisions == "Request Feedback")
        
        # Precision and Recall for deployed predictions
        deployed_mask = (agent_decisions == "Deploy")
        if np.sum(deployed_mask) > 0:
            deployed_accuracy = np.mean(model_correct[deployed_mask])
        else:
            deployed_accuracy = 0.0
        
        # True Negative Rate (correctly identifying bad predictions)
        if total_model_errors > 0:
            correct_rejections = np.sum((agent_decisions == "Request Feedback") & ~model_correct)
            true_negative_rate = correct_rejections / total_model_errors
        else:
            true_negative_rate = 1.0  # No errors to catch
        
        return {
            "Deployment Decision Accuracy": decision_accuracy,
            "False Positive Rate (FPR)": fpr,
            "True Negative Rate (TNR)": true_negative_rate,
            "Total Model Accuracy": np.mean(model_correct),
            "Total Deployed": total_deployed,
            "Total Requested": total_requested,
            "Deployed Accuracy": deployed_accuracy,
        }

    def detailed_analysis(self, confidences, predictions, true_labels, agent_decisions):
        """
        Provide detailed analysis of agent performance across confidence ranges.
        """
        confidences = np.array(confidences)
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        agent_decisions = np.array(agent_decisions)
        
        print(f"\n=== Detailed Agent Analysis ===")
        print(f"Calibrated threshold: {self.calibrated_threshold:.3f}")
        
        # Analyze performance by confidence bins
        bins = np.linspace(0, 1, 11)  # 10 bins
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        print(f"\nPerformance by confidence range:")
        print(f"{'Range':<15} {'Count':<6} {'Accuracy':<9} {'Deployed%':<10} {'Decision'}")
        print("-" * 55)
        
        for i in range(len(bins) - 1):
            mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
            if np.sum(mask) == 0:
                continue
            
            bin_count = np.sum(mask)
            bin_accuracy = np.mean((predictions == true_labels)[mask])
            bin_deployed = np.mean((agent_decisions == "Deploy")[mask])
            
            range_str = f"[{bins[i]:.1f},{bins[i+1]:.1f})"
            decision = "Deploy" if bin_centers[i] >= self.calibrated_threshold else "Request"
            
            print(f"{range_str:<15} {bin_count:<6} {bin_accuracy:<9.2%} {bin_deployed:<10.2%} {decision}")
        
        # Summary statistics
        deployed_mask = (agent_decisions == "Deploy")
        requested_mask = (agent_decisions == "Request Feedback")
        
        if np.sum(deployed_mask) > 0:
            print(f"\nDeployed predictions summary:")
            print(f"  Count: {np.sum(deployed_mask)}")
            print(f"  Accuracy: {np.mean((predictions == true_labels)[deployed_mask]):.2%}")
            print(f"  Avg confidence: {np.mean(confidences[deployed_mask]):.3f}")
            print(f"  Min confidence: {np.min(confidences[deployed_mask]):.3f}")
        
        if np.sum(requested_mask) > 0:
            print(f"\nRequested feedback summary:")
            print(f"  Count: {np.sum(requested_mask)}")
            print(f"  Accuracy: {np.mean((predictions == true_labels)[requested_mask]):.2%}")
            print(f"  Avg confidence: {np.mean(confidences[requested_mask]):.3f}")
            print(f"  Max confidence: {np.max(confidences[requested_mask]):.3f}")
