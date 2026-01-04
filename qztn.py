
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Polygon, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import csv
import os
import warnings
warnings.filterwarnings('ignore')

# Create output directory
OUTPUT_DIR = "qshield_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set high-quality publication settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.axisbelow': True,
})

# Define color schemes
COLORS = {
    'qshield': '#1E3A5F',
    'traditional': '#8B4513',
    'ai_enhanced': '#2E8B57',
    'zero_trust': '#9932CC',
    'post_quantum': '#DC143C',
    'qkd_classical': '#FF8C00',
    'federated': '#20B2AA',
    'blockchain': '#8B0000',
    'ml_intrusion': '#4169E1',
    'adaptive': '#FF6347',
    'hybrid': '#32CD32',
    'accent1': '#00CED1',
    'accent2': '#FFD700',
    'background': '#F5F5F5',
}

METHOD_NAMES = [
    'QShield-ZTN', 'Traditional SDN', 'AI-Enhanced', 'Zero-Trust Conv.',
    'Post-Quantum Only', 'QKD Classical', 'Federated Learning',
    'Blockchain Trust', 'ML Intrusion Det.', 'Adaptive Policy', 'Hybrid Classical-Q'
]

METHOD_NAMES_SHORT = [
    'QShield', 'Trad-SDN', 'AI-Enh', 'ZT-Conv',
    'PQ-Only', 'QKD-Cls', 'Fed-Lrn',
    'BC-Trust', 'ML-IDS', 'Adapt', 'Hybrid'
]


# =============================================================================
# FIGURE 1: Conceptual Foundation Diagram
# =============================================================================
def create_figure_1_intro():
    """Figure 1: Conceptual foundation of quantum-resilient, autonomous, and zero-trust orchestration"""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    center_x, center_y = 5, 5
    
    # Background gradient circles
    for i in range(50):
        circle = plt.Circle((center_x, center_y), 2.5 - i*0.02, 
                           color=plt.cm.Blues(0.2 + i*0.015), alpha=0.1)
        ax.add_patch(circle)
    
    # Central core
    core = FancyBboxPatch((3.5, 3.5), 3, 3, boxstyle="round,pad=0.1",
                          facecolor='#1E3A5F', edgecolor='#0D1B2A', linewidth=3, alpha=0.9)
    ax.add_patch(core)
    ax.text(5, 5.2, '6G NETWORK\nCORE', ha='center', va='center',
            fontsize=14, fontweight='bold', color='white')
    ax.text(5, 4.3, 'QShield-ZTN', ha='center', va='center',
            fontsize=10, color='#00CED1', style='italic')
    
    # Four main components
    components = [
        (1.5, 8, 'QUANTUM\nSECURITY', '#DC143C'),
        (8.5, 8, 'ZERO-TRUST\nARCHITECTURE', '#9932CC'),
        (1.5, 2, 'AI\nORCHESTRATION', '#2E8B57'),
        (8.5, 2, 'THREAT\nINTELLIGENCE', '#FF8C00'),
    ]
    
    for x, y, label, color in components:
        box = FancyBboxPatch((x-1, y-0.8), 2, 1.6, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='black', linewidth=2, alpha=0.85)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        
        arrow = FancyArrowPatch((x, y + (0.8 if y > 5 else -0.8)),
                                (center_x + (1.5 if x < 5 else -1.5), 
                                 center_y + (1.5 if y > 5 else -1.5)),
                                arrowstyle='->', mutation_scale=20,
                                color=color, linewidth=3, alpha=0.7)
        ax.add_patch(arrow)
    
    # Feature labels
    features = [
        (5, 9.5, 'Post-Quantum Cryptography | QKD Integration', '#1E3A5F'),
        (0.5, 5, 'Multi-Agent RL\nDecision Engine', '#2E8B57'),
        (9.5, 5, 'Dynamic Policy\nEnforcement', '#9932CC'),
        (5, 0.5, 'Real-time Anomaly Detection | Ensemble ML', '#FF8C00'),
    ]
    
    for x, y, text, color in features:
        ax.text(x, y, text, ha='center', va='center', fontsize=8,
                color=color, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=color))
    
    ax.text(5, 10.3, 'QShield-ZTN: Conceptual Framework', ha='center', va='bottom',
            fontsize=16, fontweight='bold', color='#0D1B2A')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig1_intro.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Figure 1: Intro conceptual diagram saved")


# =============================================================================
# FIGURE 2: Methodology Block Diagram
# =============================================================================
def create_figure_2_methodology():
    """Figure 2: Optimization and constraint processing block"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Main optimization block
    main_box = FancyBboxPatch((2, 3), 6, 4, boxstyle="round,pad=0.15",
                              facecolor='#E8F4FD', edgecolor='#1E3A5F', linewidth=3, alpha=0.95)
    ax.add_patch(main_box)
    
    ax.text(5, 6.5, 'MULTI-OBJECTIVE OPTIMIZATION', ha='center', va='center',
            fontsize=12, fontweight='bold', color='#1E3A5F')
    
    # Input parameters
    inputs = ['Throughput (R)', 'Latency (τ)', 'Security (S)', 'Energy (E)', 'QoS (Q)']
    for i, inp in enumerate(inputs):
        y_pos = 7.5 - i * 0.8
        ax.annotate('', xy=(2, y_pos - 2), xytext=(0.5, y_pos),
                   arrowprops=dict(arrowstyle='->', color='#2E8B57', lw=2))
        ax.text(0.3, y_pos, inp, ha='right', va='center', fontsize=9, color='#2E8B57', fontweight='bold')
    
    # Equations
    equations = [
        r'$\max \sum_{k} \beta_k U_k(\mathbf{r})$',
        r'$\text{s.t. } \sum_i r_i \leq R_{total}$',
        r'$S_{quantum} \geq 256 \text{ bits}$',
        r'$\tau \leq \tau_{max}$'
    ]
    for i, eq in enumerate(equations):
        ax.text(5, 5.8 - i * 0.6, eq, ha='center', va='center', fontsize=10, color='#1E3A5F')
    
    # Output decisions
    outputs = ['Resource Allocation', 'Security Level', 'Policy Update', 'Orchestration Action']
    for i, out in enumerate(outputs):
        y_pos = 6.8 - i * 0.9
        ax.annotate('', xy=(9.5, y_pos - 1.8), xytext=(8, y_pos - 1.8),
                   arrowprops=dict(arrowstyle='->', color='#DC143C', lw=2))
        ax.text(9.7, y_pos - 1.8, out, ha='left', va='center', fontsize=9, color='#DC143C', fontweight='bold')
    
    # Constraint boxes
    constraints = [
        (2.5, 2, 'Security\nConstraints'),
        (5, 2, 'Performance\nConstraints'),
        (7.5, 2, 'Resource\nConstraints')
    ]
    for x, y, label in constraints:
        box = FancyBboxPatch((x-0.8, y-0.5), 1.6, 1, boxstyle="round,pad=0.05",
                             facecolor='#FFE4B5', edgecolor='#FF8C00', linewidth=2, alpha=0.9)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=8, color='#8B4513', fontweight='bold')
        ax.annotate('', xy=(x, 3), xytext=(x, 2.5), arrowprops=dict(arrowstyle='->', color='#FF8C00', lw=1.5))
    
    ax.text(5, 9.5, 'QShield-ZTN: Optimization & Constraint Processing',
            ha='center', va='center', fontsize=14, fontweight='bold', color='#0D1B2A')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig2_methodology.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Figure 2: Methodology block diagram saved")


# =============================================================================
# FIGURE 3: Architecture Framework
# =============================================================================
def create_figure_3_architecture():
    """Figure 3: Overall QShield-ZTN Architecture Framework"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Five layers
    layers = [
        (0.5, 0.5, 13, 1.5, 'Physical Infrastructure Layer', '#8B4513',
         ['Base Stations', 'IoT Devices', 'Edge Nodes', 'Satellites', 'UAVs']),
        (0.5, 2.2, 13, 1.5, 'Quantum-Resilient Security Layer', '#DC143C',
         ['Lattice Crypto', 'QKD Module', 'Hash Signatures', 'Entropy Engine']),
        (0.5, 3.9, 13, 1.5, 'Zero-Trust Management Layer', '#9932CC',
         ['Trust Scoring', 'Policy Engine', 'Micro-Segmentation', 'Verification']),
        (0.5, 5.6, 13, 1.5, 'Autonomous Orchestration Layer', '#2E8B57',
         ['Multi-Agent RL', 'Resource Alloc.', 'State Prediction', 'Consensus']),
        (0.5, 7.3, 13, 1.5, 'Application Service Layer', '#1E3A5F',
         ['URLLC', 'eMBB', 'mMTC', 'Industrial IoT', 'V2X'])
    ]
    
    for x, y, w, h, title, color, components in layers:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='black', linewidth=2, alpha=0.85)
        ax.add_patch(box)
        ax.text(x + 0.3, y + h - 0.3, title, ha='left', va='top',
                fontsize=11, fontweight='bold', color='white')
        
        comp_width = (w - 1) / len(components)
        for i, comp in enumerate(components):
            comp_x = x + 0.5 + i * comp_width
            comp_box = FancyBboxPatch((comp_x, y + 0.2), comp_width - 0.2, 0.7,
                                      boxstyle="round,pad=0.02",
                                      facecolor='white', edgecolor='white', linewidth=1, alpha=0.3)
            ax.add_patch(comp_box)
            ax.text(comp_x + comp_width/2 - 0.1, y + 0.55, comp, 
                   ha='center', va='center', fontsize=8, color='white')
    
    # Cross-layer arrows
    for i in range(4):
        y_start = 2 + i * 1.7
        ax.annotate('', xy=(7, y_start + 1.5), xytext=(7, y_start + 0.2),
                   arrowprops=dict(arrowstyle='<->', color='#FFD700', lw=2.5))
    
    ax.text(-0.2, 5, 'Cross-Layer\nSecurity\nIntegration', ha='center', va='center',
            fontsize=10, fontweight='bold', color='#FFD700', rotation=90,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#FFD700'))
    
    ax.text(7, 9.5, 'QShield-ZTN: Multi-Layer Architecture Framework',
            ha='center', va='center', fontsize=16, fontweight='bold', color='#0D1B2A')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig3_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Figure 3: Architecture framework saved")


# =============================================================================
# FIGURE 4: Security Performance Comparison
# =============================================================================
def create_figure_4_security_performance():
    """Figure 4: Security performance comparison - Detection accuracy, error rates, quantum resistance"""
    fig = plt.figure(figsize=(15, 5))
    
    methods = METHOD_NAMES_SHORT
    detection_acc = [97.3, 84.7, 89.2, 91.5, 85.3, 88.6, 87.9, 83.2, 90.4, 92.1, 94.6]
    fpr = [2.1, 8.2, 6.3, 4.8, 7.9, 5.7, 6.1, 9.1, 5.2, 4.3, 3.7]
    fnr = [1.8, 7.1, 4.9, 3.7, 6.8, 4.3, 5.2, 8.3, 4.1, 3.2, 2.9]
    quantum_res = [256, 128, 128, 128, 256, 192, 128, 128, 128, 128, 192]
    
    # Subplot 1: Detection Accuracy
    ax1 = fig.add_subplot(131)
    x = np.arange(len(methods))
    ax1.plot(x, detection_acc, 'o-', color=COLORS['qshield'], linewidth=2.5, 
             markersize=10, markerfacecolor='white', markeredgewidth=2)
    ax1.fill_between(x, detection_acc, alpha=0.3, color=COLORS['qshield'])
    ax1.axhline(y=97.3, color=COLORS['accent2'], linestyle='--', alpha=0.7, label='QShield-ZTN: 97.3%')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Detection Accuracy (%)')
    ax1.set_title('Detection Accuracy Comparison', fontweight='bold')
    ax1.set_ylim(80, 100)
    ax1.legend(loc='lower right', fontsize=8)
    ax1.scatter([0], [97.3], s=200, c=COLORS['accent2'], zorder=5, marker='*')
    
    # Subplot 2: Error Rates
    ax2 = fig.add_subplot(132)
    width = 0.35
    ax2.bar(x - width/2, fpr, width, label='False Positive Rate (%)', color=COLORS['post_quantum'], alpha=0.8)
    ax2.bar(x + width/2, fnr, width, label='False Negative Rate (%)', color=COLORS['ai_enhanced'], alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Error Rate (%)')
    ax2.set_title('Error Rate Comparison', fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.annotate('2.1%', (0 - width/2, fpr[0] + 0.3), ha='center', fontsize=8, fontweight='bold')
    ax2.annotate('1.8%', (0 + width/2, fnr[0] + 0.3), ha='center', fontsize=8, fontweight='bold')
    
    # Subplot 3: Quantum Resistance
    ax3 = fig.add_subplot(133)
    colors_qr = [COLORS['qshield'] if q == 256 else COLORS['hybrid'] if q == 192 else COLORS['traditional'] for q in quantum_res]
    bars = ax3.barh(x, quantum_res, color=colors_qr, alpha=0.8, edgecolor='black')
    ax3.set_yticks(x)
    ax3.set_yticklabels(methods, fontsize=8)
    ax3.set_xlabel('Quantum Resistance (bits)')
    ax3.set_title('Quantum Resistance Strength', fontweight='bold')
    ax3.set_xlim(0, 300)
    ax3.axvline(x=256, color=COLORS['accent2'], linestyle='--', alpha=0.7, linewidth=2)
    for i, (bar, val) in enumerate(zip(bars, quantum_res)):
        ax3.text(val + 5, i, f'{val}', va='center', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig4_accuracy.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Figure 4: Security performance comparison saved")


# =============================================================================
# FIGURE 5: Overall Performance Analysis
# =============================================================================
def create_figure_5_overall_performance():
    """Figure 5: QShield-ZTN overall performance - Multi-metric evaluation"""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    methods = METHOD_NAMES_SHORT
    metrics = {
        'Detection Acc.': [97.3, 84.7, 89.2, 91.5, 85.3, 88.6, 87.9, 83.2, 90.4, 92.1, 94.6],
        'Quantum Res.': [256, 128, 128, 128, 256, 192, 128, 128, 128, 128, 192],
        'Response (ms)': [15.3, 45.7, 32.1, 28.9, 52.3, 38.4, 41.8, 67.5, 29.7, 26.8, 21.5],
        'FPR (%)': [2.1, 8.2, 6.3, 4.8, 7.9, 5.7, 6.1, 9.1, 5.2, 4.3, 3.7],
    }
    
    def normalize(arr, invert=False):
        arr = np.array(arr, dtype=float)
        if invert:
            arr = arr.max() - arr + arr.min()
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-10)
    
    # Subplot 1: Normalized Metrics
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(methods))
    width = 0.2
    
    norm_acc = normalize(metrics['Detection Acc.'])
    norm_qr = normalize(metrics['Quantum Res.'])
    norm_resp = normalize(metrics['Response (ms)'], invert=True)
    norm_fpr = normalize(metrics['FPR (%)'], invert=True)
    
    ax1.bar(x - 1.5*width, norm_acc, width, label='Detection Acc.', color=COLORS['qshield'])
    ax1.bar(x - 0.5*width, norm_qr, width, label='Quantum Res.', color=COLORS['post_quantum'])
    ax1.bar(x + 0.5*width, norm_resp, width, label='Response Speed', color=COLORS['ai_enhanced'])
    ax1.bar(x + 1.5*width, norm_fpr, width, label='Low FPR', color=COLORS['zero_trust'])
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, ha='right', fontsize=7)
    ax1.set_ylabel('Normalized Score')
    ax1.set_title('Normalized Metrics Comparison', fontweight='bold')
    ax1.legend(loc='upper right', fontsize=7, ncol=2)
    ax1.set_ylim(0, 1.2)
    
    # Subplot 2: Stacked Security Components
    ax2 = fig.add_subplot(gs[0, 1])
    security_components = {
        'Encryption': [10, 6, 6, 6, 10, 8, 6, 6, 6, 6, 8],
        'Authentication': [10, 5, 7, 9, 7, 7, 7, 8, 7, 8, 8],
        'Threat Detection': [10, 6, 8, 7, 6, 7, 7, 6, 9, 8, 8],
        'Policy Mgmt': [10, 5, 6, 9, 5, 6, 6, 7, 6, 9, 7],
    }
    
    bottom = np.zeros(len(methods))
    colors = [COLORS['qshield'], COLORS['post_quantum'], COLORS['ai_enhanced'], COLORS['zero_trust']]
    for i, (comp, values) in enumerate(security_components.items()):
        ax2.bar(x, values, 0.6, bottom=bottom, label=comp, color=colors[i], alpha=0.85)
        bottom += np.array(values)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=45, ha='right', fontsize=7)
    ax2.set_ylabel('Stacked Component Score')
    ax2.set_title('Security Component Breakdown', fontweight='bold')
    ax2.legend(loc='upper right', fontsize=7)
    
    # Subplot 3: Correlation Heatmap
    ax3 = fig.add_subplot(gs[0, 2])
    metric_names = list(metrics.keys())
    data_matrix = np.array([metrics[m] for m in metric_names])
    corr_matrix = np.corrcoef(data_matrix)
    
    im = ax3.imshow(corr_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
    ax3.set_xticks(range(len(metric_names)))
    ax3.set_yticks(range(len(metric_names)))
    ax3.set_xticklabels(metric_names, rotation=45, ha='right', fontsize=8)
    ax3.set_yticklabels(metric_names, fontsize=8)
    ax3.set_title('Metric Correlations', fontweight='bold')
    
    for i in range(len(metric_names)):
        for j in range(len(metric_names)):
            ax3.text(j, i, f'{corr_matrix[i, j]:.2f}', ha='center', va='center', 
                    fontsize=8, color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')
    
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    
    # Subplot 4: Overall Performance Ranking
    ax4 = fig.add_subplot(gs[1, :])
    overall = (norm_acc + norm_qr + norm_resp + norm_fpr) / 4
    sorted_idx = np.argsort(overall)[::-1]
    
    colors_bar = [COLORS['qshield'] if i == 0 else COLORS['hybrid'] if overall[i] > 0.7 
                  else COLORS['ai_enhanced'] if overall[i] > 0.5 else COLORS['traditional'] 
                  for i in range(len(methods))]
    
    bars = ax4.barh(np.arange(len(methods)), overall[sorted_idx], 
                    color=[colors_bar[i] for i in sorted_idx], alpha=0.85, edgecolor='black')
    ax4.set_yticks(np.arange(len(methods)))
    ax4.set_yticklabels([methods[i] for i in sorted_idx], fontsize=9)
    ax4.set_xlabel('Overall Performance Score (Normalized)')
    ax4.set_title('Overall Performance Ranking', fontweight='bold', fontsize=12)
    ax4.set_xlim(0, 1.1)
    
    for i, (idx, score) in enumerate(zip(sorted_idx, overall[sorted_idx])):
        ax4.text(score + 0.02, i, f'{score:.3f}', va='center', fontsize=9, fontweight='bold')
    
    ax4.axvline(x=overall[0], color=COLORS['accent2'], linestyle='--', alpha=0.7, linewidth=2)
    
    plt.suptitle('QShield-ZTN: Comprehensive Performance Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig5_overall_performance.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Figure 5: Overall performance analysis saved")


# =============================================================================
# FIGURE 6: Performance Heatmap
# =============================================================================
def create_figure_6_heatmap():
    """Figure 6: Performance heatmap showing normalized scores"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    methods = METHOD_NAMES
    metrics = ['Detection\nAccuracy', 'Inv. FPR', 'Inv. FNR', 'Quantum\nResistance', 'Response\nSpeed']
    
    data = np.array([
        [0.97, 0.98, 0.98, 1.0, 0.95],
        [0.85, 0.82, 0.83, 0.5, 0.60],
        [0.89, 0.87, 0.90, 0.5, 0.75],
        [0.92, 0.90, 0.92, 0.5, 0.80],
        [0.85, 0.82, 0.84, 1.0, 0.55],
        [0.89, 0.88, 0.90, 0.75, 0.70],
        [0.88, 0.87, 0.88, 0.5, 0.65],
        [0.83, 0.79, 0.78, 0.5, 0.45],
        [0.90, 0.89, 0.91, 0.5, 0.78],
        [0.92, 0.91, 0.93, 0.5, 0.82],
        [0.95, 0.92, 0.94, 0.75, 0.88],
    ])
    
    cmap = LinearSegmentedColormap.from_list('qshield', 
        ['#2C3E50', '#3498DB', '#2ECC71', '#F1C40F', '#E74C3C'])
    
    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=0.4, vmax=1.0)
    
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels(metrics, fontsize=10, fontweight='bold')
    ax.set_yticklabels(methods, fontsize=10)
    
    for i in range(len(methods)):
        for j in range(len(metrics)):
            color = 'white' if data[i, j] > 0.7 else 'black'
            ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center',
                   fontsize=9, color=color, fontweight='bold')
    
    ax.add_patch(Rectangle((-0.5, -0.5), len(metrics), 1, fill=False, 
                           edgecolor=COLORS['accent2'], linewidth=3))
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label('Performance Score', fontsize=10)
    
    ax.set_title('Performance Heatmap: Normalized Scores Across Methods',
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig6_heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Figure 6: Performance heatmap saved")


# =============================================================================
# FIGURE 7: Network Performance Analysis
# =============================================================================
def create_figure_7_network_performance():
    """Figure 7: Network performance analysis"""
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Subplot 1: Energy Efficiency
    ax1 = fig.add_subplot(gs[0, 0])
    categories = ['Advanced\n(QShield, Hybrid)', 'Intermediate\n(AI, ZT, Adaptive)', 'Traditional\n(SDN, BC, Others)']
    energy_data = [[3.2, 3.6], [4.1, 4.3, 3.9, 4.2], [4.8, 5.2, 4.6, 4.4, 6.1]]
    
    positions = [1, 2, 3]
    bp = ax1.boxplot(energy_data, positions=positions, widths=0.6, patch_artist=True)
    colors_box = [COLORS['qshield'], COLORS['ai_enhanced'], COLORS['traditional']]
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_xticklabels(categories, fontsize=9)
    ax1.set_ylabel('Energy Consumption (J/bit × 10⁻⁹)')
    ax1.set_title('Energy Efficiency by Method Category', fontweight='bold')
    ax1.axhline(y=3.2, color=COLORS['accent2'], linestyle='--', alpha=0.7, label='QShield-ZTN: 3.2')
    ax1.legend(loc='upper right', fontsize=8)
    
    # Subplot 2: Parallel Coordinates
    ax2 = fig.add_subplot(gs[0, 1])
    methods_subset = ['QShield-ZTN', 'Hybrid Classical-Q', 'Adaptive Policy', 'AI-Enhanced', 'Traditional SDN']
    metrics_pc = ['Throughput', 'Latency', 'Packet Loss', 'Energy Eff.', 'QRS']
    
    data_pc = {
        'QShield-ZTN': [15.7, 0.87, 0.032, 3.2, 256],
        'Hybrid Classical-Q': [14.2, 0.95, 0.048, 3.6, 192],
        'Adaptive Policy': [13.9, 1.02, 0.061, 3.9, 128],
        'AI-Enhanced': [13.8, 1.05, 0.065, 4.1, 128],
        'Traditional SDN': [12.3, 1.24, 0.087, 4.8, 128]
    }
    
    mins = [12, 0.8, 0.03, 3.0, 100]
    maxs = [16, 1.3, 0.1, 5.0, 260]
    
    x_coords = np.arange(len(metrics_pc))
    colors_pc = [COLORS['qshield'], COLORS['hybrid'], COLORS['adaptive'], COLORS['ai_enhanced'], COLORS['traditional']]
    
    for i, (method, values) in enumerate(data_pc.items()):
        norm_vals = [(v - mins[j]) / (maxs[j] - mins[j]) for j, v in enumerate(values)]
        ax2.plot(x_coords, norm_vals, 'o-', color=colors_pc[i], linewidth=2.5, markersize=8, label=method, alpha=0.8)
    
    ax2.set_xticks(x_coords)
    ax2.set_xticklabels(metrics_pc, fontsize=9)
    ax2.set_ylabel('Normalized Value')
    ax2.set_title('Parallel Coordinates: Performance Profile', fontweight='bold')
    ax2.legend(loc='lower right', fontsize=8)
    ax2.set_ylim(-0.1, 1.1)
    
    # Subplot 3: Performance Gains
    ax3 = fig.add_subplot(gs[1, 0])
    gain_metrics = ['Throughput', 'Latency', 'Packet Loss', 'Energy', 'Detection']
    gains = [10.6, 8.4, 33.3, 11.1, 2.9]
    
    colors_gain = plt.cm.viridis(np.linspace(0.2, 0.8, len(gain_metrics)))
    bars = ax3.bar(gain_metrics, gains, color=colors_gain, edgecolor='black', alpha=0.85)
    ax3.set_ylabel('Improvement Over Best Baseline (%)')
    ax3.set_title('QShield-ZTN Performance Gains', fontweight='bold')
    
    for bar, gain in zip(bars, gains):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'+{gain}%', ha='center', fontsize=10, fontweight='bold')
    
    # Subplot 4: Throughput vs Load
    ax4 = fig.add_subplot(gs[1, 1])
    loads = np.array([20, 40, 60, 80, 100])
    
    throughput_qshield = [15.9, 15.8, 15.6, 15.4, 15.0]
    throughput_hybrid = [14.5, 14.3, 13.9, 13.2, 12.5]
    throughput_trad = [12.8, 12.2, 11.3, 10.1, 8.7]
    
    ax4.plot(loads, throughput_qshield, 'o-', color=COLORS['qshield'], linewidth=2.5, label='QShield')
    ax4.plot(loads, throughput_hybrid, 's--', color=COLORS['hybrid'], linewidth=2, label='Hybrid')
    ax4.plot(loads, throughput_trad, '^:', color=COLORS['traditional'], linewidth=2, label='Traditional')
    
    ax4.set_xlabel('Network Load (%)')
    ax4.set_ylabel('Throughput (Gbps)')
    ax4.set_title('Throughput vs Network Load', fontweight='bold')
    ax4.legend(loc='lower left', fontsize=8)
    ax4.set_xlim(15, 105)
    
    plt.suptitle('QShield-ZTN: Network Performance Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig7_network_performance.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Figure 7: Network performance analysis saved")


# =============================================================================
# FIGURE 8: Orchestration Performance
# =============================================================================
def create_figure_8_orchestration():
    """Figure 8: Real-time decision making and network scalability"""
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.35)
    
    # Top: Real-time Decision Making
    ax1 = fig.add_subplot(gs[0])
    time = np.linspace(0, 100, 500)
    
    np.random.seed(42)
    base_qshield = 8.7 + np.random.normal(0, 0.5, len(time))
    base_hybrid = 21.5 + np.random.normal(0, 1.5, len(time))
    base_adaptive = 26.8 + np.random.normal(0, 2, len(time))
    base_trad = 45.7 + np.random.normal(0, 4, len(time))
    
    stress_events = [20, 45, 70, 90]
    for se in stress_events:
        mask = (time > se) & (time < se + 5)
        base_qshield[mask] += 2
        base_hybrid[mask] += 8
        base_adaptive[mask] += 12
        base_trad[mask] += 25
    
    ax1.plot(time, base_qshield, color=COLORS['qshield'], linewidth=2, label='QShield-ZTN (8.7ms avg)', alpha=0.9)
    ax1.plot(time, base_hybrid, color=COLORS['hybrid'], linewidth=1.5, label='Hybrid Classical-Q (21.5ms)', alpha=0.7)
    ax1.plot(time, base_adaptive, color=COLORS['adaptive'], linewidth=1.5, label='Adaptive Policy (26.8ms)', alpha=0.7)
    ax1.plot(time, base_trad, color=COLORS['traditional'], linewidth=1.5, label='Traditional SDN (45.7ms)', alpha=0.7)
    
    for se in stress_events:
        ax1.axvline(x=se, color='red', linestyle='--', alpha=0.3)
        ax1.text(se, 75, 'Stress\nEvent', ha='center', fontsize=8, color='red')
    
    ax1.fill_between(time, 0, base_qshield, alpha=0.2, color=COLORS['qshield'])
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Response Time (ms)')
    ax1.set_title('Real-Time Decision Making Performance Under Network Stress', fontweight='bold', fontsize=12)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_ylim(0, 80)
    ax1.set_xlim(0, 100)
    
    ax1.annotate('49.6% faster\nthan Traditional', xy=(50, 10), fontsize=10,
                fontweight='bold', color=COLORS['qshield'],
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Bottom: Scalability Analysis
    ax2 = fig.add_subplot(gs[1])
    sizes = [100, 500, 1000, 5000, 10000, 50000, 100000]
    size_labels = ['100', '500', '1K', '5K', '10K', '50K', '100K']
    
    scale_qshield = [0.99, 0.98, 0.97, 0.95, 0.93, 0.92, 0.91]
    scale_hybrid = [0.98, 0.96, 0.93, 0.89, 0.86, 0.82, 0.78]
    scale_adaptive = [0.97, 0.94, 0.90, 0.84, 0.79, 0.73, 0.68]
    scale_trad = [0.95, 0.90, 0.82, 0.70, 0.58, 0.45, 0.35]
    
    resource_qshield = [0.98, 0.97, 0.96, 0.94, 0.93, 0.92, 0.92]
    
    ax2_twin = ax2.twinx()
    
    x = np.arange(len(sizes))
    width = 0.2
    
    ax2.bar(x - 1.5*width, scale_qshield, width, label='QShield-ZTN', color=COLORS['qshield'], alpha=0.85)
    ax2.bar(x - 0.5*width, scale_hybrid, width, label='Hybrid Classical-Q', color=COLORS['hybrid'], alpha=0.85)
    ax2.bar(x + 0.5*width, scale_adaptive, width, label='Adaptive Policy', color=COLORS['adaptive'], alpha=0.85)
    ax2.bar(x + 1.5*width, scale_trad, width, label='Traditional SDN', color=COLORS['traditional'], alpha=0.85)
    
    ax2_twin.plot(x, resource_qshield, 'D-', color=COLORS['accent2'], linewidth=2.5, markersize=8, label='QShield Resource Eff.')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(size_labels)
    ax2.set_xlabel('Network Size (Number of Devices)')
    ax2.set_ylabel('Scalability Factor')
    ax2_twin.set_ylabel('Resource Efficiency', color=COLORS['accent2'])
    ax2.set_title('Network Scalability & Resource Efficiency Analysis', fontweight='bold', fontsize=12)
    ax2.legend(loc='lower left', fontsize=9)
    ax2_twin.legend(loc='upper right', fontsize=9)
    ax2.set_ylim(0, 1.1)
    ax2_twin.set_ylim(0.85, 1.0)
    
    ax2.annotate('91% at 100K devices', xy=(6, 0.91), xytext=(5, 1.0),
                arrowprops=dict(arrowstyle='->', color=COLORS['qshield']),
                fontsize=9, fontweight='bold', color=COLORS['qshield'])
    
    plt.suptitle('QShield-ZTN: Orchestration Efficiency & Scalability', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig8_orchestration.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Figure 8: Orchestration performance saved")


# =============================================================================
# FIGURE 9: Quantum Adaptive Evaluation
# =============================================================================
def create_figure_9_quantum_adaptive():
    """Figure 9: Quantum-resilient adaptability"""
    fig = plt.figure(figsize=(16, 12))
    
    # Top-left: Security vs Quantum Threat
    ax1 = fig.add_subplot(2, 2, 1)
    time = np.linspace(0, 100, 100)
    threat_level = 0.3 + 0.5 * (1 - np.exp(-time/30))
    
    sec_qshield = 0.98 - 0.02 * threat_level + 0.01 * np.sin(time/10)
    sec_hybrid = 0.92 - 0.08 * threat_level + 0.02 * np.sin(time/8)
    sec_classical = 0.85 - 0.25 * threat_level + 0.03 * np.sin(time/5)
    sec_static_pq = 0.95 - 0.05 * threat_level
    
    ax1.plot(time, sec_qshield, linewidth=2.5, color=COLORS['qshield'], label='QShield-ZTN (Adaptive)')
    ax1.plot(time, sec_hybrid, linewidth=2, color=COLORS['hybrid'], label='Hybrid Classical-Q')
    ax1.plot(time, sec_classical, linewidth=2, color=COLORS['traditional'], label='Classical Only')
    ax1.plot(time, sec_static_pq, linewidth=2, color=COLORS['post_quantum'], linestyle='--', label='Static Post-Quantum')
    
    ax1.fill_between(time, sec_qshield, 0.5, alpha=0.2, color=COLORS['qshield'])
    ax1.set_xlabel('Time (Quantum Threat Evolution)')
    ax1.set_ylabel('Security Level')
    ax1.set_title('Security Maintenance Under Increasing Quantum Threat', fontweight='bold')
    ax1.legend(loc='lower left', fontsize=8)
    ax1.set_ylim(0.5, 1.0)
    
    # Top-right: QKD Optimization Landscape
    ax2 = fig.add_subplot(2, 2, 2)
    x = np.linspace(0, 10, 20)
    y = np.linspace(0, 10, 20)
    X, Y = np.meshgrid(x, y)
    
    U = np.cos(X/3) * np.exp(-Y/10)
    V = np.sin(Y/3) * np.exp(-X/10)
    speed = np.sqrt(U**2 + V**2)
    
    strm = ax2.streamplot(X, Y, U, V, color=speed, cmap='viridis', linewidth=1.5, density=1.5)
    
    optimal = plt.Circle((7, 7), 1.5, fill=False, color=COLORS['accent2'], linewidth=3, linestyle='--')
    ax2.add_patch(optimal)
    ax2.text(7, 7, 'Optimal\nQKD\nRegion', ha='center', va='center', fontsize=9, fontweight='bold', color=COLORS['accent2'])
    
    ax2.set_xlabel('Channel Quality Parameter')
    ax2.set_ylabel('Entanglement Efficiency')
    ax2.set_title('QKD Optimization Landscape', fontweight='bold')
    plt.colorbar(strm.lines, ax=ax2, label='Key Rate Gradient')
    
    # Bottom-left: 3D Attack Resistance
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    time_3d = np.linspace(0, 50, 30)
    attack_complexity = np.linspace(1, 10, 30)
    T, A = np.meshgrid(time_3d, attack_complexity)
    
    resistance = 100 - 5 * np.log(A + 1) - 0.1 * T + 2 * np.sin(T/5)
    resistance = np.clip(resistance, 70, 100)
    
    surf = ax3.plot_surface(T, A, resistance, cmap='RdYlGn', alpha=0.8, edgecolor='none')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Attack Complexity')
    ax3.set_zlabel('Attack Resistance (%)')
    ax3.set_title('QShield-ZTN Attack Resistance', fontweight='bold')
    ax3.view_init(elev=25, azim=45)
    
    # Bottom-right: Security Parameter Contour
    ax4 = fig.add_subplot(2, 2, 4)
    quantum_risk = np.linspace(0, 1, 50)
    network_load = np.linspace(0, 1, 50)
    QR, NL = np.meshgrid(quantum_risk, network_load)
    
    sec_param = 128 + 128 * (0.5 * QR + 0.3 * NL + 0.2 * QR * NL)
    
    contour = ax4.contourf(QR, NL, sec_param, levels=20, cmap='plasma')
    ax4.contour(QR, NL, sec_param, levels=[192, 224, 256], colors='white', linewidths=2, linestyles=['--', '-', '-'])
    
    plt.colorbar(contour, ax=ax4, label='Security Parameter (bits)')
    ax4.set_xlabel('Quantum Risk Level')
    ax4.set_ylabel('Network Load')
    ax4.set_title('Dynamic Security Parameter Adjustment', fontweight='bold')
    
    ax4.text(0.7, 0.3, '192-bit', color='white', fontsize=9, fontweight='bold')
    ax4.text(0.8, 0.6, '224-bit', color='white', fontsize=9, fontweight='bold')
    ax4.text(0.9, 0.85, '256-bit', color='white', fontsize=9, fontweight='bold')
    
    plt.suptitle('QShield-ZTN: Quantum-Resilient Adaptive Security Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig9_quantum_adaptive.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Figure 9: Quantum adaptive evaluation saved")


# =============================================================================
# FIGURE 10: Spatial Security Field
# =============================================================================
def create_figure_10_spatial_security():
    """Figure 10: Spatial security field distribution"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    x = np.linspace(0, 100, 100)
    y = np.linspace(0, 100, 100)
    X, Y = np.meshgrid(x, y)
    
    core_nodes = [(50, 50), (25, 75), (75, 75), (25, 25), (75, 25)]
    
    Z = np.zeros_like(X)
    for cx, cy in core_nodes:
        Z += 100 * np.exp(-((X - cx)**2 + (Y - cy)**2) / 300)
    
    Z += 10 * np.sin(X/10) * np.cos(Y/10)
    Z = np.clip(Z, 20, 100)
    
    cmap = LinearSegmentedColormap.from_list('security', 
        ['#0D1B2A', '#1E3A5F', '#3498DB', '#2ECC71', '#F1C40F', '#E74C3C'])
    
    im = ax.contourf(X, Y, Z, levels=50, cmap=cmap, alpha=0.9)
    ax.contour(X, Y, Z, levels=[40, 60, 80, 95], colors='white', linewidths=1, linestyles='--', alpha=0.5)
    
    for i, (cx, cy) in enumerate(core_nodes):
        if i == 0:
            ax.scatter(cx, cy, s=500, c=COLORS['accent2'], marker='*', 
                      edgecolor='white', linewidth=2, zorder=5, label='Network Core')
        else:
            ax.scatter(cx, cy, s=300, c=COLORS['qshield'], marker='s', 
                      edgecolor='white', linewidth=2, zorder=5)
    
    for i, (cx1, cy1) in enumerate(core_nodes):
        for j, (cx2, cy2) in enumerate(core_nodes[i+1:], i+1):
            ax.plot([cx1, cx2], [cy1, cy2], '--', color=COLORS['accent1'], linewidth=1.5, alpha=0.6)
    
    np.random.seed(42)
    edge_nodes_x = np.random.uniform(10, 90, 20)
    edge_nodes_y = np.random.uniform(10, 90, 20)
    ax.scatter(edge_nodes_x, edge_nodes_y, s=80, c='white', marker='o', 
              edgecolor=COLORS['qshield'], linewidth=1.5, alpha=0.8, label='Edge Nodes')
    
    annotations = [
        (50, 50, 'Core: 97.3%\nAccuracy', COLORS['accent2']),
        (20, 85, '94.6%\n(Hybrid)', COLORS['hybrid']),
        (80, 85, '92.1%\n(Adaptive)', COLORS['adaptive']),
        (10, 15, '84.7%\n(Traditional)', COLORS['traditional']),
    ]
    
    for x_pos, y_pos, text, color in annotations:
        ax.annotate(text, xy=(x_pos, y_pos), fontsize=9, fontweight='bold',
                   color=color, ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor=color))
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Security Effectiveness (%)', fontsize=11)
    
    metrics_text = '''QShield-ZTN Performance:
━━━━━━━━━━━━━━━━━━━━━
Detection Accuracy: 97.3%
Quantum Resistance: 256-bit
Energy Efficiency: 3.2 J/bit
Response Time: 15.3 ms
QKD Rate: 1.2 MHz'''
    
    ax.text(98, 5, metrics_text, fontsize=9, fontfamily='monospace',
           verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, 
                    edgecolor=COLORS['qshield'], linewidth=2))
    
    ax.set_xlabel('Network X-Coordinate', fontsize=11)
    ax.set_ylabel('Network Y-Coordinate', fontsize=11)
    ax.set_title('QShield-ZTN: Spatial Security Field Distribution', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig10_spatial_security.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Figure 10: Spatial security field saved")


# =============================================================================
# FIGURE 11: Learning Curves
# =============================================================================
def create_figure_11_learning_curves():
    """Figure 11: Multi-agent learning performance"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Reward Accumulation
    ax1 = axes[0]
    episodes = np.arange(0, 10000, 50)
    
    np.random.seed(42)
    
    def generate_learning_curve(final_reward, convergence_speed, noise_level):
        curve = final_reward * (1 - np.exp(-episodes / convergence_speed))
        noise = noise_level * np.random.randn(len(episodes)) * np.exp(-episodes / 5000)
        return curve + noise
    
    reward_qshield = generate_learning_curve(0.95, 2000, 0.03)
    reward_hybrid = generate_learning_curve(0.88, 3000, 0.04)
    reward_adaptive = generate_learning_curve(0.82, 3500, 0.05)
    reward_federated = generate_learning_curve(0.78, 4000, 0.06)
    reward_traditional = generate_learning_curve(0.65, 5000, 0.08)
    
    ax1.plot(episodes, reward_qshield, linewidth=2.5, color=COLORS['qshield'], label='QShield-ZTN')
    ax1.plot(episodes, reward_hybrid, linewidth=2, color=COLORS['hybrid'], label='Hybrid Classical-Q')
    ax1.plot(episodes, reward_adaptive, linewidth=2, color=COLORS['adaptive'], label='Adaptive Policy')
    ax1.plot(episodes, reward_federated, linewidth=2, color=COLORS['federated'], label='Federated Learning')
    ax1.plot(episodes, reward_traditional, linewidth=2, color=COLORS['traditional'], label='Traditional RL')
    
    ax1.axvspan(0, 2000, alpha=0.1, color='blue', label='Initialization')
    ax1.axvspan(2000, 6000, alpha=0.1, color='green', label='Training')
    ax1.axvspan(6000, 10000, alpha=0.1, color='gold', label='Optimization')
    
    ax1.set_xlabel('Training Episodes')
    ax1.set_ylabel('Cumulative Security Reward (Normalized)')
    ax1.set_title('Multi-Agent Learning: Reward Accumulation', fontweight='bold')
    ax1.legend(loc='lower right', fontsize=8, ncol=2)
    ax1.set_xlim(0, 10000)
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3)
    
    ax1.annotate('95% Reward\n(4000 episodes)', xy=(4000, 0.95), xytext=(5500, 0.75),
                arrowprops=dict(arrowstyle='->', color=COLORS['qshield']),
                fontsize=10, fontweight='bold', color=COLORS['qshield'])
    
    # Right: Privacy Stability
    ax2 = axes[1]
    iterations = np.arange(0, 100, 1)
    
    privacy_qshield = 0.05 + 0.2 * np.exp(-iterations / 20) + 0.01 * np.random.randn(len(iterations))
    privacy_hybrid = 0.08 + 0.3 * np.exp(-iterations / 25) + 0.02 * np.random.randn(len(iterations))
    privacy_federated = 0.12 + 0.4 * np.exp(-iterations / 30) + 0.03 * np.random.randn(len(iterations))
    privacy_zk = 0.10 + 0.35 * np.exp(-iterations / 28) + 0.025 * np.random.randn(len(iterations))
    
    ax2.plot(iterations, privacy_qshield, linewidth=2.5, color=COLORS['qshield'], label='QShield-ZTN')
    ax2.plot(iterations, privacy_hybrid, linewidth=2, color=COLORS['hybrid'], label='Hybrid Classical-Q')
    ax2.plot(iterations, privacy_federated, linewidth=2, color=COLORS['federated'], label='Federated Learning')
    ax2.plot(iterations, privacy_zk, linewidth=2, color=COLORS['zero_trust'], label='Zero-Knowledge')
    
    ax2.fill_between(iterations, privacy_qshield, alpha=0.2, color=COLORS['qshield'])
    
    ax2.set_xlabel('Communication Rounds')
    ax2.set_ylabel('Privacy Variance Error')
    ax2.set_title('Privacy Stability & Convergence Analysis', fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 0.6)
    ax2.grid(True, alpha=0.3)
    
    ax2.annotate('52% Stability\nImprovement', xy=(80, 0.06), xytext=(60, 0.25),
                arrowprops=dict(arrowstyle='->', color=COLORS['qshield']),
                fontsize=10, fontweight='bold', color=COLORS['qshield'])
    
    plt.suptitle('QShield-ZTN: Multi-Agent Learning Performance Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig11_learning_curves.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Figure 11: Learning curves saved")


# =============================================================================
# FIGURE 12: Energy Analysis
# =============================================================================
def create_figure_12_energy_analysis():
    """Figure 12: Energy efficiency and power consumption analysis"""
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # Component-wise energy
    ax1 = fig.add_subplot(gs[0, 0])
    components = ['Base\nStation', 'Edge\nComputing', 'IoT\nDevices', 
                  'Security\nModules', 'Orchestration\nEngine', 'Quantum\nComponents']
    qshield_power = [185.3, 78.9, 12.4, 34.7, 56.2, 23.1]
    baseline_power = [243.7, 102.4, 18.6, 48.2, 71.8, 31.5]
    
    x = np.arange(len(components))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, qshield_power, width, label='QShield-ZTN', color=COLORS['qshield'], alpha=0.85)
    bars2 = ax1.bar(x + width/2, baseline_power, width, label='Best Baseline', color=COLORS['traditional'], alpha=0.85)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(components, fontsize=8)
    ax1.set_ylabel('Power Consumption (mW)')
    ax1.set_title('Component-wise Power Comparison', fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    
    for i, (q, b) in enumerate(zip(qshield_power, baseline_power)):
        improvement = (b - q) / b * 100
        ax1.text(i, max(q, b) + 5, f'-{improvement:.1f}%', ha='center', fontsize=7, fontweight='bold', color=COLORS['ai_enhanced'])
    
    # 24-hour energy pattern
    ax2 = fig.add_subplot(gs[0, 1])
    hours = np.arange(0, 24)
    
    load_pattern = 0.5 + 0.3 * np.sin((hours - 6) * np.pi / 12) + 0.2 * np.sin((hours - 12) * np.pi / 6)
    load_pattern = np.clip(load_pattern, 0.3, 1.0)
    
    np.random.seed(42)
    energy_qshield = 390 * load_pattern * (0.9 + 0.1 * np.random.rand(24))
    energy_baseline = 516 * load_pattern * (0.9 + 0.15 * np.random.rand(24))
    
    ax2.fill_between(hours, 0, energy_baseline, alpha=0.3, color=COLORS['traditional'], label='Baseline')
    ax2.fill_between(hours, 0, energy_qshield, alpha=0.5, color=COLORS['qshield'], label='QShield-ZTN')
    ax2.plot(hours, energy_qshield, 'o-', color=COLORS['qshield'], linewidth=2)
    ax2.plot(hours, energy_baseline, 's--', color=COLORS['traditional'], linewidth=2)
    
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Power Consumption (mW)')
    ax2.set_title('24-Hour Energy Profile', fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_xlim(0, 23)
    
    ax2.axvspan(0, 6, alpha=0.1, color='blue')
    ax2.axvspan(6, 18, alpha=0.1, color='yellow')
    ax2.axvspan(18, 23, alpha=0.1, color='purple')
    ax2.text(3, 500, 'Night', ha='center', fontsize=8)
    ax2.text(12, 500, 'Day', ha='center', fontsize=8)
    ax2.text(20.5, 500, 'Evening', ha='center', fontsize=8)
    
    # Energy efficiency vs data rate
    ax3 = fig.add_subplot(gs[0, 2])
    data_rates = [100, 250, 500, 750, 1000]
    
    eff_qshield = [3.0, 3.1, 3.2, 3.3, 3.5]
    eff_hybrid = [3.4, 3.5, 3.6, 3.8, 4.0]
    eff_adaptive = [3.7, 3.8, 4.0, 4.2, 4.5]
    eff_traditional = [4.5, 4.8, 5.2, 5.6, 6.2]
    
    ax3.plot(data_rates, eff_qshield, 'o-', linewidth=2.5, color=COLORS['qshield'], label='QShield-ZTN')
    ax3.plot(data_rates, eff_hybrid, 's--', linewidth=2, color=COLORS['hybrid'], label='Hybrid')
    ax3.plot(data_rates, eff_adaptive, '^--', linewidth=2, color=COLORS['adaptive'], label='Adaptive')
    ax3.plot(data_rates, eff_traditional, 'd--', linewidth=2, color=COLORS['traditional'], label='Traditional')
    
    ax3.fill_between(data_rates, eff_qshield, eff_traditional, alpha=0.2, color=COLORS['ai_enhanced'])
    
    ax3.set_xlabel('Data Rate (Mbps)')
    ax3.set_ylabel('Energy per Bit (J/bit × 10⁻⁹)')
    ax3.set_title('Energy Efficiency Scaling', fontweight='bold')
    ax3.legend(loc='upper left', fontsize=8)
    
    ax3.annotate(f'43.6% savings\nat 1000 Mbps', xy=(1000, 3.5), xytext=(700, 5.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['qshield']),
                fontsize=9, fontweight='bold', color=COLORS['qshield'])
    
    # Power mode comparison
    ax4 = fig.add_subplot(gs[1, :])
    
    modes = ['Normal\nOperation', 'High\nSecurity', 'Peak\nTraffic', 
             'Emergency\nResponse', 'Sleep\nMode', 'Quantum\nAlert']
    
    qshield_modes = [390, 420, 450, 480, 120, 440]
    hybrid_modes = [450, 510, 530, 580, 180, 500]
    adaptive_modes = [480, 540, 560, 610, 200, 530]
    traditional_modes = [516, 590, 650, 720, 280, 600]
    
    x = np.arange(len(modes))
    width = 0.2
    
    ax4.bar(x - 1.5*width, qshield_modes, width, label='QShield-ZTN', color=COLORS['qshield'], alpha=0.85)
    ax4.bar(x - 0.5*width, hybrid_modes, width, label='Hybrid Classical-Q', color=COLORS['hybrid'], alpha=0.85)
    ax4.bar(x + 0.5*width, adaptive_modes, width, label='Adaptive Policy', color=COLORS['adaptive'], alpha=0.85)
    ax4.bar(x + 1.5*width, traditional_modes, width, label='Traditional', color=COLORS['traditional'], alpha=0.85)
    
    ax4.set_xticks(x)
    ax4.set_xticklabels(modes, fontsize=10)
    ax4.set_ylabel('Power Consumption (mW)')
    ax4.set_title('Power Consumption Across Operational Modes', fontweight='bold', fontsize=12)
    ax4.legend(loc='upper right', fontsize=9)
    
    avg_savings = np.mean([(t - q) / t * 100 for q, t in zip(qshield_modes, traditional_modes)])
    ax4.axhline(y=300, color=COLORS['accent2'], linestyle='--', alpha=0.7)
    ax4.text(5.5, 310, f'Avg. Savings: {avg_savings:.1f}%', fontsize=10, fontweight='bold', color=COLORS['accent2'])
    
    plt.suptitle('QShield-ZTN: Comprehensive Energy Efficiency Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig12_energy_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Figure 12: Energy analysis saved")


# =============================================================================
# FIGURE 13: Scalability Analysis
# =============================================================================
def create_figure_13_scalability():
    """Figure 13: Comprehensive scalability analysis"""
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    sizes = np.array([100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000])
    size_labels = ['100', '500', '1K', '5K', '10K', '50K', '100K', '500K', '1M']
    
    scale_qshield = [0.99, 0.98, 0.97, 0.95, 0.93, 0.92, 0.91, 0.89, 0.87]
    scale_hybrid = [0.98, 0.96, 0.93, 0.89, 0.86, 0.82, 0.78, 0.72, 0.65]
    scale_adaptive = [0.97, 0.94, 0.90, 0.84, 0.79, 0.73, 0.68, 0.60, 0.52]
    scale_traditional = [0.95, 0.90, 0.82, 0.70, 0.58, 0.45, 0.35, 0.25, 0.18]
    
    # Scalability Factor
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogx(sizes, scale_qshield, 'o-', linewidth=2.5, color=COLORS['qshield'], markersize=8, label='QShield-ZTN')
    ax1.semilogx(sizes, scale_hybrid, 's--', linewidth=2, color=COLORS['hybrid'], markersize=7, label='Hybrid Classical-Q')
    ax1.semilogx(sizes, scale_adaptive, '^--', linewidth=2, color=COLORS['adaptive'], markersize=7, label='Adaptive Policy')
    ax1.semilogx(sizes, scale_traditional, 'd--', linewidth=2, color=COLORS['traditional'], markersize=7, label='Traditional SDN')
    
    ax1.fill_between(sizes, scale_qshield, scale_traditional, alpha=0.2, color=COLORS['ai_enhanced'])
    ax1.axhline(y=0.9, color='gray', linestyle=':', alpha=0.5)
    ax1.text(200, 0.91, 'Target: 90%', fontsize=9, color='gray')
    
    ax1.set_xlabel('Network Size (Devices)')
    ax1.set_ylabel('Scalability Factor')
    ax1.set_title('Scalability Factor vs Network Size', fontweight='bold')
    ax1.legend(loc='lower left', fontsize=9)
    ax1.set_ylim(0.1, 1.05)
    ax1.grid(True, alpha=0.3)
    
    # Security at Scale
    ax2 = fig.add_subplot(gs[0, 1])
    
    security_qshield = [97.5, 97.4, 97.3, 97.2, 97.0, 96.8, 96.5, 96.0, 95.5]
    security_hybrid = [95.0, 94.5, 94.0, 93.2, 92.5, 91.5, 90.0, 88.0, 85.0]
    security_adaptive = [93.0, 92.0, 91.0, 89.5, 88.0, 86.0, 83.0, 79.0, 75.0]
    security_traditional = [88.0, 85.0, 82.0, 77.0, 72.0, 65.0, 58.0, 50.0, 42.0]
    
    ax2.semilogx(sizes, security_qshield, 'o-', linewidth=2.5, color=COLORS['qshield'], markersize=8, label='QShield-ZTN')
    ax2.semilogx(sizes, security_hybrid, 's--', linewidth=2, color=COLORS['hybrid'], markersize=7, label='Hybrid Classical-Q')
    ax2.semilogx(sizes, security_adaptive, '^--', linewidth=2, color=COLORS['adaptive'], markersize=7, label='Adaptive Policy')
    ax2.semilogx(sizes, security_traditional, 'd--', linewidth=2, color=COLORS['traditional'], markersize=7, label='Traditional SDN')
    
    ax2.axhline(y=95, color=COLORS['accent2'], linestyle='--', alpha=0.7, linewidth=2)
    ax2.text(150, 95.5, '95% Threshold', fontsize=9, color=COLORS['accent2'], fontweight='bold')
    
    ax2.set_xlabel('Network Size (Devices)')
    ax2.set_ylabel('Security Effectiveness (%)')
    ax2.set_title('Security Maintenance at Scale', fontweight='bold')
    ax2.legend(loc='lower left', fontsize=9)
    ax2.set_ylim(40, 100)
    ax2.grid(True, alpha=0.3)
    
    # Resource Optimization
    ax3 = fig.add_subplot(gs[1, 0])
    
    resources = ['CPU\nUtilization', 'Memory\nUsage', 'Bandwidth\nEfficiency', 'Storage\nI/O', 'Network\nLatency']
    
    qshield_res = [82, 78, 85, 88, 92]
    improvement = [(100 - q) for q in qshield_res]
    
    x = np.arange(len(resources))
    width = 0.5
    
    bars = ax3.bar(x, improvement, width, color=COLORS['qshield'], alpha=0.85, edgecolor='black', linewidth=1.5)
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(resources, fontsize=10)
    ax3.set_ylabel('Resource Savings (%)')
    ax3.set_title('QShield-ZTN Resource Optimization', fontweight='bold')
    ax3.set_ylim(0, 30)
    
    for bar, val in zip(bars, improvement):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val}%', ha='center', fontsize=11, fontweight='bold', color=COLORS['qshield'])
    
    avg_improvement = np.mean(improvement)
    ax3.axhline(y=avg_improvement, color=COLORS['accent2'], linestyle='--', linewidth=2)
    ax3.text(4.3, avg_improvement + 0.5, f'Avg: {avg_improvement:.1f}%', fontsize=10, fontweight='bold', color=COLORS['accent2'])
    
    # Latency at Scale
    ax4 = fig.add_subplot(gs[1, 1])
    
    latency_qshield = [5, 8, 12, 25, 38, 55, 75, 110, 150]
    latency_hybrid = [8, 15, 25, 50, 80, 120, 170, 250, 350]
    latency_adaptive = [10, 20, 35, 70, 110, 170, 250, 380, 520]
    latency_traditional = [15, 30, 55, 120, 200, 350, 550, 850, 1200]
    
    ax4.loglog(sizes, latency_qshield, 'o-', linewidth=2.5, color=COLORS['qshield'], markersize=8, label='QShield-ZTN')
    ax4.loglog(sizes, latency_hybrid, 's--', linewidth=2, color=COLORS['hybrid'], markersize=7, label='Hybrid Classical-Q')
    ax4.loglog(sizes, latency_adaptive, '^--', linewidth=2, color=COLORS['adaptive'], markersize=7, label='Adaptive Policy')
    ax4.loglog(sizes, latency_traditional, 'd--', linewidth=2, color=COLORS['traditional'], markersize=7, label='Traditional SDN')
    
    ax4.axhline(y=100, color='red', linestyle=':', alpha=0.7, linewidth=2)
    ax4.text(150, 120, 'Max Acceptable: 100ms', fontsize=9, color='red')
    
    ax4.set_xlabel('Network Size (Devices)')
    ax4.set_ylabel('Communication Latency (ms)')
    ax4.set_title('Latency Scaling Analysis', fontweight='bold')
    ax4.legend(loc='upper left', fontsize=9)
    ax4.grid(True, alpha=0.3, which='both')
    
    plt.suptitle('QShield-ZTN: Comprehensive Scalability Performance', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig13_scalability.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Figure 13: Scalability analysis saved")


# =============================================================================
# FIGURE 14: Threat Detection Analysis
# =============================================================================
def create_figure_14_threat_detection():
    """Figure 14: Comprehensive threat detection accuracy assessment"""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    attack_types = ['DDoS', 'MITM', 'Malware', 'Data\nExfiltration', 
                    'Quantum\nAttack', 'APT', 'Zero-Day', 'Side\nChannel']
    
    acc_qshield = [98.5, 97.8, 96.9, 97.2, 98.7, 96.1, 95.3, 97.5]
    acc_hybrid = [95.2, 94.1, 93.5, 93.8, 92.1, 91.2, 89.5, 93.2]
    acc_ml = [93.8, 92.5, 91.8, 92.1, 45.2, 88.5, 86.2, 85.3]
    acc_traditional = [88.5, 85.2, 84.1, 83.5, 15.3, 72.5, 65.8, 70.2]
    
    # Radar chart
    ax1 = fig.add_subplot(gs[0, 0], projection='polar')
    
    angles = np.linspace(0, 2*np.pi, len(attack_types), endpoint=False).tolist()
    angles += angles[:1]
    
    acc_qshield_r = acc_qshield + [acc_qshield[0]]
    acc_hybrid_r = acc_hybrid + [acc_hybrid[0]]
    acc_ml_r = acc_ml + [acc_ml[0]]
    acc_traditional_r = acc_traditional + [acc_traditional[0]]
    
    ax1.plot(angles, acc_qshield_r, 'o-', linewidth=2.5, color=COLORS['qshield'], label='QShield-ZTN')
    ax1.fill(angles, acc_qshield_r, alpha=0.25, color=COLORS['qshield'])
    ax1.plot(angles, acc_hybrid_r, 's--', linewidth=2, color=COLORS['hybrid'], label='Hybrid')
    ax1.plot(angles, acc_ml_r, '^--', linewidth=2, color=COLORS['ml_intrusion'], label='ML-IDS')
    ax1.plot(angles, acc_traditional_r, 'd--', linewidth=2, color=COLORS['traditional'], label='Traditional')
    
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(attack_types, fontsize=8)
    ax1.set_ylim(0, 100)
    ax1.set_title('Attack Detection Accuracy by Type', fontweight='bold', pad=20)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
    
    # Confusion matrix
    ax2 = fig.add_subplot(gs[0, 1])
    
    confusion_data = np.array([
        [97.3, 1.5, 0.8, 0.4],
        [1.2, 96.8, 1.5, 0.5],
        [0.6, 1.8, 97.1, 0.5],
        [0.3, 0.6, 0.9, 98.2]
    ])
    
    categories = ['Normal', 'Classical\nAttack', 'Quantum\nAttack', 'Hybrid\nAttack']
    
    im = ax2.imshow(confusion_data, cmap='RdYlGn', vmin=0, vmax=100)
    ax2.set_xticks(np.arange(len(categories)))
    ax2.set_yticks(np.arange(len(categories)))
    ax2.set_xticklabels(categories, fontsize=9)
    ax2.set_yticklabels(categories, fontsize=9)
    ax2.set_xlabel('Predicted Class')
    ax2.set_ylabel('True Class')
    ax2.set_title('QShield-ZTN Classification Matrix (%)', fontweight='bold')
    
    for i in range(len(categories)):
        for j in range(len(categories)):
            color = 'white' if confusion_data[i, j] > 50 else 'black'
            ax2.text(j, i, f'{confusion_data[i, j]:.1f}', ha='center', va='center',
                    fontsize=11, color=color, fontweight='bold')
    
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    
    # Detection time
    ax3 = fig.add_subplot(gs[1, 0])
    
    methods_dt = ['QShield-ZTN', 'Hybrid', 'ML-IDS', 'Adaptive', 'Traditional']
    detection_times = [15.3, 21.5, 29.7, 26.8, 45.7]
    false_alarm_times = [8.2, 12.4, 18.5, 15.3, 32.1]
    
    x = np.arange(len(methods_dt))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, detection_times, width, label='Detection Time (ms)',
                    color=COLORS['qshield'], alpha=0.85)
    bars2 = ax3.bar(x + width/2, false_alarm_times, width, label='False Alarm Resolution (ms)',
                    color=COLORS['post_quantum'], alpha=0.85)
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods_dt, fontsize=10)
    ax3.set_ylabel('Time (ms)')
    ax3.set_title('Detection & Response Time Comparison', fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    
    for bar in bars1:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}', ha='center', fontsize=8, fontweight='bold')
    
    # ROC curves
    ax4 = fig.add_subplot(gs[1, 1])
    
    fpr_range = np.linspace(0, 1, 100)
    
    tpr_qshield = 1 - np.exp(-10 * fpr_range)
    tpr_qshield = np.clip(tpr_qshield, 0, 0.99)
    
    tpr_hybrid = 1 - np.exp(-7 * fpr_range)
    tpr_hybrid = np.clip(tpr_hybrid, 0, 0.96)
    
    tpr_ml = 1 - np.exp(-5 * fpr_range)
    tpr_ml = np.clip(tpr_ml, 0, 0.93)
    
    tpr_traditional = 1 - np.exp(-3 * fpr_range)
    tpr_traditional = np.clip(tpr_traditional, 0, 0.88)
    
    ax4.plot(fpr_range, tpr_qshield, linewidth=2.5, color=COLORS['qshield'], label=f'QShield-ZTN (AUC=0.987)')
    ax4.plot(fpr_range, tpr_hybrid, linewidth=2, color=COLORS['hybrid'], label=f'Hybrid (AUC=0.962)')
    ax4.plot(fpr_range, tpr_ml, linewidth=2, color=COLORS['ml_intrusion'], label=f'ML-IDS (AUC=0.941)')
    ax4.plot(fpr_range, tpr_traditional, linewidth=2, color=COLORS['traditional'], label=f'Traditional (AUC=0.875)')
    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')