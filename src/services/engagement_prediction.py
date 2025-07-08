from src.models.domain import (
    EngagementPrediction, EngagementLevel
)

def generate_engagement_prediction(features) -> EngagementPrediction:
    """
    Generate engagement prediction from extracted features.
    
    This is a simplified prediction algorithm. In production, this would
    use a trained ML model with proper feature engineering.
    """
    text_features = features.text_features
    image_features = features.image_features
    
    # Base score from text sentiment
    base_score = 50.0  # Neutral baseline
    
    # Sentiment contribution (30% weight)
    sentiment_contribution = text_features.sentiment_score * 15.0
    
    # Readability contribution (20% weight)
    readability_contribution = (text_features.readability_score - 50) * 0.2
    
    # Content length contribution (15% weight)
    optimal_length = 100  # Optimal word count
    length_factor = 1.0 - abs(text_features.word_count - optimal_length) / optimal_length
    length_contribution = length_factor * 7.5
    
    # Social signals contribution (20% weight)
    social_signals = (
        len(text_features.hashtags) * 2 +
        len(text_features.mentions) * 1.5 +
        text_features.emoji_count * 1
    )
    social_contribution = min(social_signals, 10.0)
    
    # Image contribution (15% weight)
    image_contribution = 0.0
    if image_features:
        # Good aspect ratio
        if 0.8 <= image_features.aspect_ratio <= 1.2:
            image_contribution += 5.0
        
        # Text-image alignment
        if features.text_image_similarity:
            image_contribution += features.text_image_similarity * 10.0
    
    # Calculate final score
    final_score = (
        base_score +
        sentiment_contribution +
        readability_contribution +
        length_contribution +
        social_contribution +
        image_contribution
    )
    
    # Ensure score is within bounds
    final_score = max(0.0, min(100.0, final_score))
    
    # Calculate confidence based on feature quality
    confidence_factors = [
        min(text_features.word_count / 50.0, 1.0),  # Text length
        (text_features.readability_score / 100.0),  # Readability
        features.content_coherence,  # Content coherence
    ]
    
    if image_features:
        confidence_factors.append(0.8)  # Image presence
    
    confidence = sum(confidence_factors) / len(confidence_factors)
    confidence = max(0.1, min(1.0, confidence))
    
    # Determine engagement level
    if final_score >= 80:
        level = EngagementLevel.VERY_HIGH
    elif final_score >= 65:
        level = EngagementLevel.HIGH
    elif final_score >= 45:
        level = EngagementLevel.MEDIUM
    else:
        level = EngagementLevel.LOW
    
    # Generate contributing factors
    factors = {
        "sentiment": sentiment_contribution / 15.0,
        "readability": readability_contribution / 10.0,
        "content_length": length_contribution / 7.5,
        "social_signals": social_contribution / 10.0,
        "image_quality": image_contribution / 15.0 if image_features else 0.0
    }
    
    # Generate recommendations
    recommendations = _generate_recommendations(text_features, image_features, final_score)
    
    return EngagementPrediction(
        score=final_score,
        confidence=confidence,
        level=level,
        factors=factors,
        recommendations=recommendations
    )


def _generate_recommendations(text_features, image_features, score: float) -> list[str]:
    """Generate actionable recommendations for content improvement."""
    recommendations = []
    
    # Text recommendations
    if text_features.sentiment_score < 0.1:
        recommendations.append("Consider adding more positive language to improve sentiment")
    
    if text_features.word_count < 20:
        recommendations.append("Add more descriptive content to increase engagement")
    elif text_features.word_count > 200:
        recommendations.append("Consider shortening the text for better readability")
    
    if text_features.readability_score < 50:
        recommendations.append("Simplify language and use shorter sentences for better readability")
    
    if len(text_features.hashtags) == 0:
        recommendations.append("Add relevant hashtags to increase discoverability")
    elif len(text_features.hashtags) > 5:
        recommendations.append("Reduce number of hashtags to avoid appearing spammy")
    
    if text_features.emoji_count == 0:
        recommendations.append("Consider adding emojis to make content more engaging")
    
    # Image recommendations
    if not image_features:
        recommendations.append("Add a high-quality image to significantly boost engagement")
    else:
        if image_features.aspect_ratio < 0.8 or image_features.aspect_ratio > 1.2:
            recommendations.append("Use images with better aspect ratios (close to square)")
    
    # Overall score recommendations
    if score < 50:
        recommendations.append("Focus on improving content quality and visual appeal")
    elif score < 70:
        recommendations.append("Good foundation - fine-tune sentiment and add visual elements")
    
    return recommendations[:5]  # Limit to top 5 recommendations