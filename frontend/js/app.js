const API_URL = 'http://localhost:8000';

const emotionMap = {
    'joy': { pt: 'Alegria', emoji: '😄', color: 'var(--color-joy)' },
    'sadness': { pt: 'Tristeza', emoji: '😢', color: 'var(--color-sadness)' },
    'anger': { pt: 'Raiva', emoji: '😡', color: 'var(--color-anger)' },
    'fear': { pt: 'Medo', emoji: '😨', color: 'var(--color-fear)' },
    'surprise': { pt: 'Surpresa', emoji: '😲', color: 'var(--color-surprise)' },
    'disgust': { pt: 'Nojo', emoji: '🤢', color: 'var(--color-disgust)' }
};

document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('emotion-form');
    const textarea = document.getElementById('text-input');
    const charCount = document.getElementById('char-count');
    const analyzeBtn = document.getElementById('analyze-btn');
    const btnText = document.querySelector('.btn-text');
    const loader = document.getElementById('btn-loader');
    const resultsSection = document.getElementById('results-section');
    const errorToast = document.getElementById('error-message');

    // Update char count
    textarea.addEventListener('input', () => {
        const count = textarea.value.length;
        charCount.textContent = `${count} / 5000`;
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const text = textarea.value.trim();
        if (text.length < 2) return;

        // UI Loading state
        setLoading(true);
        errorToast.classList.add('hidden');
        resultsSection.classList.add('hidden');

        try {
            const response = await fetch(`${API_URL}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ texto: text })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'Erro ao comunicar com a API');
            }

            renderResults(data);
        } catch (error) {
            showError(error.message);
        } finally {
            setLoading(false);
        }
    });

    function setLoading(isLoading) {
        if (isLoading) {
            btnText.classList.add('hidden');
            loader.classList.remove('hidden');
            analyzeBtn.disabled = true;
            textarea.disabled = true;
        } else {
            btnText.classList.remove('hidden');
            loader.classList.add('hidden');
            analyzeBtn.disabled = false;
            textarea.disabled = false;
        }
    }

    function showError(message) {
        errorToast.textContent = message;
        errorToast.classList.remove('hidden');
    }

    function renderResults(data) {
        // Render Detected Emotions
        const detectedContainer = document.getElementById('detected-emotions-container');
        detectedContainer.innerHTML = '';
        
        let primaryColor = '#ffffff';

        data.emocoes_detectadas.forEach((item, index) => {
            const emotionData = emotionMap[item.emocao] || { pt: 'Neutro', emoji: '😐', color: '#9ca3af' };
            const confidencePct = Math.round(item.confianca * 100);
            
            if (index === 0) {
                primaryColor = emotionData.color;
            }

            const tag = document.createElement('div');
            tag.className = 'emotion-tag';
            tag.style.display = 'inline-flex';
            tag.style.alignItems = 'center';
            tag.style.gap = '8px';
            tag.style.padding = '8px 16px';
            tag.style.borderRadius = '20px';
            tag.style.border = `1px solid ${emotionData.color}`;
            tag.style.background = 'rgba(255, 255, 255, 0.05)';
            
            tag.innerHTML = `
                <span style="font-size: 1.2rem;">${emotionData.emoji}</span>
                <span style="color: ${emotionData.color}; font-weight: 600;">${emotionData.pt}</span>
                <span style="background: rgba(255,255,255,0.1); padding: 2px 8px; border-radius: 12px; font-size: 0.8rem; margin-left: 5px;">${confidencePct}%</span>
            `;
            
            detectedContainer.appendChild(tag);
        });

        // Set glow based on most confident emotion
        const primaryCard = document.getElementById('primary-result-card');
        primaryCard.style.boxShadow = `0 10px 40px -10px ${primaryColor}40`; // 40 is hex for 25% opacity

        // Render Probabilities Bars
        const barsContainer = document.getElementById('bars-container');
        barsContainer.innerHTML = '';

        // Sort probabilities highest to lowest
        const sortedProbs = Object.entries(data.probabilidades)
            .sort(([, a], [, b]) => b - a);

        sortedProbs.forEach(([key, prob]) => {
            const pct = Math.round(prob * 100);
            const emotion = emotionMap[key];

            const barRow = document.createElement('div');
            barRow.className = 'bar-row';
            barRow.innerHTML = `
                <div class="bar-label">
                    <span>${emotion.emoji}</span>
                    <span>${emotion.pt}</span>
                </div>
                <div class="bar-track">
                    <div class="bar-fill" style="background-color: ${emotion.color}; width: 0%"></div>
                </div>
                <div class="bar-value">${pct}%</div>
            `;
            
            barsContainer.appendChild(barRow);
            
            // Trigger animation after append
            setTimeout(() => {
                const fill = barRow.querySelector('.bar-fill');
                fill.style.width = `${pct}%`;
            }, 50);
        });

        // Show section
        resultsSection.classList.remove('hidden');
        
        // Smooth scroll to results
        setTimeout(() => {
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }, 100);
    }
});
