// 火災偵測系統 - 增強視覺效果 JavaScript

// 初始化粒子背景效果
function initParticleBackground() {
    const particleContainer = document.createElement('div');
    particleContainer.className = 'particles-bg';
    document.body.appendChild(particleContainer);
    
    // 創建粒子
    for (let i = 0; i < 50; i++) {
        createParticle(particleContainer);
    }
}

function createParticle(container) {
    const particle = document.createElement('div');
    particle.className = 'particle';
    
    // 隨機位置和動畫延遲
    particle.style.left = Math.random() * 100 + '%';
    particle.style.animationDelay = Math.random() * 20 + 's';
    particle.style.animationDuration = (15 + Math.random() * 10) + 's';
    
    container.appendChild(particle);
    
    // 粒子動畫結束後重新創建
    particle.addEventListener('animationend', () => {
        particle.remove();
        createParticle(container);
    });
}

// 火災按鈕震動效果
function addFireButtonShake() {
    const fireButtons = document.querySelectorAll('.fire-detected');
    fireButtons.forEach(button => {
        button.addEventListener('mouseenter', () => {
            button.style.animation = 'shake 0.5s ease-in-out';
        });
        
        button.addEventListener('animationend', () => {
            button.style.animation = '';
        });
    });
}

// 添加震動動畫CSS
const shakeCSS = `
@keyframes shake {
    0%, 20%, 40%, 60%, 80%, 100% { transform: translateX(0); }
    10%, 30%, 50%, 70%, 90% { transform: translateX(-3px); }
}
`;

// 動態添加CSS
function addDynamicCSS(cssText) {
    const style = document.createElement('style');
    style.textContent = cssText;
    document.head.appendChild(style);
}

// 信心度圓環進度條動畫
function createConfidenceCircle(container, percentage, color, label) {
    const size = 120;
    const strokeWidth = 8;
    const radius = (size - strokeWidth) / 2;
    const circumference = radius * 2 * Math.PI;
    const offset = circumference - (percentage / 100) * circumference;
    
    container.innerHTML = `
        <div class="confidence-circle">
            <svg width="${size}" height="${size}">
                <circle class="circle-bg" 
                        cx="${size/2}" cy="${size/2}" r="${radius}"/>
                <circle class="circle-progress" 
                        cx="${size/2}" cy="${size/2}" r="${radius}"
                        stroke="${color}"
                        stroke-dasharray="${circumference}"
                        stroke-dashoffset="${offset}"/>
            </svg>
            <div class="circle-text" style="color: ${color}">
                ${percentage}%<br>
                <small style="font-size: 0.8rem;">${label}</small>
            </div>
        </div>
    `;
    
    // 動畫效果
    const progressCircle = container.querySelector('.circle-progress');
    setTimeout(() => {
        progressCircle.style.strokeDashoffset = offset;
    }, 500);
}

// 打字機效果
function typewriterEffect(element, text, speed = 50) {
    element.innerHTML = '';
    element.style.width = '0';
    element.classList.add('typewriter');
    
    let i = 0;
    const timer = setInterval(() => {
        if (i < text.length) {
            element.innerHTML += text.charAt(i);
            i++;
        } else {
            clearInterval(timer);
            element.classList.remove('typewriter');
            element.style.width = 'auto';
        }
    }, speed);
}

// 滾動動畫觸發器
function initScrollAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.animationPlayState = 'running';
                entry.target.classList.add('fade-in');
            }
        });
    }, observerOptions);
    
    // 觀察所有需要動畫的元素
    document.querySelectorAll('.action-item, .confidence-card, .accordion-item').forEach(el => {
        el.style.animationPlayState = 'paused';
        observer.observe(el);
    });
}

// 音效管理器
class SoundManager {
    constructor() {
        this.sounds = {};
        this.enabled = false;
    }
    
    async loadSound(name, url) {
        try {
            const audio = new Audio(url);
            audio.preload = 'auto';
            this.sounds[name] = audio;
        } catch (error) {
            console.log('音效載入失敗:', error);
        }
    }
    
    play(name, volume = 0.5) {
        if (this.enabled && this.sounds[name]) {
            this.sounds[name].volume = volume;
            this.sounds[name].currentTime = 0;
            this.sounds[name].play().catch(() => {
                // 靜默處理音效播放失敗
            });
        }
    }
    
    enable() {
        this.enabled = true;
    }
    
    disable() {
        this.enabled = false;
    }
}

// 全域音效管理器
const soundManager = new SoundManager();

// 火災警報聲效果（使用Web Audio API模擬）
function createFireAlarmSound() {
    if (!window.AudioContext && !window.webkitAudioContext) return;
    
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    
    function playAlarmBeep() {
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();
        
        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);
        
        oscillator.frequency.setValueAtTime(800, audioContext.currentTime);
        oscillator.frequency.setValueAtTime(1000, audioContext.currentTime + 0.1);
        
        gainNode.gain.setValueAtTime(0, audioContext.currentTime);
        gainNode.gain.linearRampToValueAtTime(0.1, audioContext.currentTime + 0.01);
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.2);
        
        oscillator.start(audioContext.currentTime);
        oscillator.stop(audioContext.currentTime + 0.2);
    }
    
    return playAlarmBeep;
}

// 成功提示音效果
function createSuccessSound() {
    if (!window.AudioContext && !window.webkitAudioContext) return;
    
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    
    function playSuccessSound() {
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();
        
        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);
        
        oscillator.frequency.setValueAtTime(523.25, audioContext.currentTime); // C5
        oscillator.frequency.setValueAtTime(659.25, audioContext.currentTime + 0.1); // E5
        oscillator.frequency.setValueAtTime(783.99, audioContext.currentTime + 0.2); // G5
        
        gainNode.gain.setValueAtTime(0, audioContext.currentTime);
        gainNode.gain.linearRampToValueAtTime(0.1, audioContext.currentTime + 0.01);
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.3);
        
        oscillator.start(audioContext.currentTime);
        oscillator.stop(audioContext.currentTime + 0.3);
    }
    
    return playSuccessSound;
}

// 觸覺反饋（震動）
function addHapticFeedback(element, pattern = [100]) {
    if ('vibrate' in navigator) {
        element.addEventListener('click', () => {
            navigator.vibrate(pattern);
        });
    }
}

// 鍵盤快捷鍵支援
function initKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        // Ctrl/Cmd + U: 快速上傳
        if ((e.ctrlKey || e.metaKey) && e.key === 'u') {
            e.preventDefault();
            document.getElementById('file').click();
        }
        
        // Escape: 關閉預覽
        if (e.key === 'Escape') {
            const previewContainer = document.getElementById('previewContainer');
            if (previewContainer.style.display === 'block') {
                removePreview();
            }
        }
        
        // Enter: 提交表單（當焦點在表單元素上時）
        if (e.key === 'Enter' && e.target.closest('form')) {
            const submitBtn = document.getElementById('submitBtn');
            if (!submitBtn.disabled) {
                e.preventDefault();
                submitBtn.click();
            }
        }
    });
}

// 自動儲存功能
function initAutoSave() {
    const roleSelect = document.getElementById('role');
    
    // 載入上次選擇的角色
    const savedRole = localStorage.getItem('fire-detection-role');
    if (savedRole && roleSelect) {
        roleSelect.value = savedRole;
    }
    
    // 儲存角色選擇
    if (roleSelect) {
        roleSelect.addEventListener('change', () => {
            localStorage.setItem('fire-detection-role', roleSelect.value);
        });
    }
}

// 初始化所有增強效果
function initEnhancedEffects() {
    // 添加動態CSS
    addDynamicCSS(shakeCSS);
    
    // 初始化各種效果
    initParticleBackground();
    initScrollAnimations();
    initKeyboardShortcuts();
    initAutoSave();
    
    // 創建音效
    const fireAlarmSound = createFireAlarmSound();
    const successSound = createSuccessSound();
    
    // 為按鈕添加音效和觸覺反饋
    document.addEventListener('click', (e) => {
        if (e.target.matches('.fire-detected')) {
            if (fireAlarmSound) fireAlarmSound();
            addHapticFeedback(e.target, [50, 50, 50]);
        } else if (e.target.matches('.no-fire-detected')) {
            if (successSound) successSound();
            addHapticFeedback(e.target, [100]);
        }
    });
    
    // 監聽結果顯示事件
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            mutation.addedNodes.forEach((node) => {
                if (node.nodeType === 1 && node.classList.contains('result-card')) {
                    addFireButtonShake();
                    
                    // 添加結果出現動畫
                    node.classList.add('result-appear');
                    
                    // 為信心度卡片添加圓環效果
                    const confidenceCards = node.querySelectorAll('.confidence-card');
                    confidenceCards.forEach((card, index) => {
                        const valueElement = card.querySelector('.confidence-value');
                        const percentage = parseInt(valueElement.textContent);
                        const color = valueElement.classList.contains('fire') ? '#e74c3c' : '#2ecc71';
                        const label = card.querySelector('.text-muted').textContent;
                        
                        setTimeout(() => {
                            createConfidenceCircle(card, percentage, color, label);
                        }, index * 200);
                    });
                }
            });
        });
    });
    
    observer.observe(document.querySelector('.result-container'), {
        childList: true,
        subtree: true
    });
}

// 當DOM載入完成時初始化
document.addEventListener('DOMContentLoaded', initEnhancedEffects);

// 導出給全域使用
window.enhancedEffects = {
    soundManager,
    typewriterEffect,
    createConfidenceCircle,
    addHapticFeedback
}; 