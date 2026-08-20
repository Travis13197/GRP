#!/usr/bin/env python3
"""
Phase L1: Kabsch Alignment + Intrinsic Covariance + Ledoit-Wolf Shrinkage
==========================================================================

Law 1 v2 核心修复 — 解决缺口 A1 (主管线无Kabsch对齐, 刚体模式混入内禀涨落)

核心功能:
  1. kabsch_align_ensemble(): 将系综中所有构象对齐到公共参考系 (去SE(3)刚体自由度)
  2. ledoit_wolf_shrinkage(): Ledoit-Wolf 最优收缩估计协方差 (闭式λ*)
  3. compute_intrinsic_metric(): 完整管线 — Kabsch对齐 → 内禀协方差 → LW收缩 → g_S

物理意义:
  - 原始协方差在实验室系计算, 整体转动/平动混入内禀涨落
  - 对 PR≈3 的 IDP, 前3主成分可能就是刚体模式 (M9审计)
  - Kabsch对齐后, 协方差仅反映内禀构象涨落, 满足 SE(3)-不变性
  - 对齐后应有6个近零特征值 (3平动+3转动), λ_{3N-5..3N}≈0

作者: ProtGenesis2 Ensemble
日期: 2026-07-19
"""

import numpy as np
from typing import Dict, Tuple, Optional
import warnings


def kabsch_align(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Kabsch 算法: 将点集 P 对齐到点集 Q (最小二乘旋转).

    参数:
        P: (N, 3) 待对齐坐标
        Q: (N, 3) 参考坐标

    返回:
        P_aligned: (N, 3) 对齐后的坐标
        R: (3, 3) 最优旋转矩阵
    """
    # 中心化
    P_cent = P - P.mean(axis=0)
    Q_cent = Q - Q.mean(axis=0)

    # 协方差矩阵
    H = P_cent.T @ Q_cent

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # 防止反射 (保证 det(R) = +1)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])

    # 最优旋转
    R = Vt.T @ D @ U.T

    # 应用旋转
    P_aligned = P_cent @ R.T + Q.mean(axis=0)

    return P_aligned, R


def kabsch_align_ensemble(positions: np.ndarray,
                          reference: Optional[np.ndarray] = None,
                          reference_mode: str = 'first') -> np.ndarray:
    """
    将系综中所有构象通过 Kabsch 算法对齐到公共参考系.

    参数:
        positions: (n_samples, n_residues, 3) 构象坐标系综
        reference: (n_residues, 3) 可选自定义参考构象. 若为None, 使用 reference_mode
        reference_mode: 'first' (第一帧), 'mean' (平均构象), 'medoid' (中心构象)

    返回:
        aligned: (n_samples, n_residues, 3) 对齐后的构象坐标系综

    物理意义:
        去除 SE(3) 刚体自由度 (3平动 + 3转动), 使协方差仅反映内禀构象涨落.
        对齐后, 系综协方差矩阵应有6个近零特征值 (对应刚体模式).
    """
    n_samples, n_residues, _ = positions.shape

    # 选择参考构象
    if reference is not None:
        ref = reference
    elif reference_mode == 'first':
        ref = positions[0]
    elif reference_mode == 'mean':
        # 迭代对齐到平均构象 (2轮通常足够)
        ref = positions[0].copy()
        for _ in range(2):
            aligned_temp = np.zeros_like(positions)
            for i in range(n_samples):
                aligned_temp[i], _ = kabsch_align(positions[i], ref)
            ref = aligned_temp.mean(axis=0)
    elif reference_mode == 'medoid':
        # 选择到所有其他构象 RMSD 最小的构象
        rmsd_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                p_aligned, _ = kabsch_align(positions[i], positions[j])
                rmsd = np.sqrt(np.mean((p_aligned - positions[j]) ** 2))
                rmsd_matrix[i, j] = rmsd
                rmsd_matrix[j, i] = rmsd
        medoid_idx = np.argmin(rmsd_matrix.sum(axis=1))
        ref = positions[medoid_idx]
    else:
        raise ValueError(f"Unknown reference_mode: {reference_mode}")

    # 对齐所有构象
    aligned = np.zeros_like(positions)
    for i in range(n_samples):
        aligned[i], _ = kabsch_align(positions[i], ref)

    return aligned


def ledoit_wolf_shrinkage(X: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Ledoit-Wolf 最优收缩协方差估计 (闭式解).

    估计: C_shrunk = (1-λ*)·S + λ*·(tr(S)/q)·I
    其中 S 为样本协方差, q 为维度, λ*∈[0,1] 为最优收缩强度.

    参数:
        X: (n_samples, n_features) 数据中心化后的矩阵

    返回:
        C_shrunk: (n_features, n_features) 收缩后的协方差矩阵
        lambda_star: 最优收缩强度 λ*

    物理意义:
        有限采样下, 样本协方差的特征值谱系统性展宽 (小特征值被低估, 大特征值被高估).
        LW收缩用闭式解析解找到偏差-方差权衡的最优点, 替换人为常数 ε=0.01·tr(C)/q.
    """
    n, q = X.shape

    # 样本协方差 (无偏)
    S = (X.T @ X) / (n - 1)

    # 目标: 各向同性缩放单位矩阵
    mu = np.trace(S) / q
    T = mu * np.eye(q)

    # 计算最优收缩强度 λ* (Ledoit-Wolf 2004 闭式解)
    # λ* = min(1, max(0, (Σ||x_i x_i^T - S||²_F) / (n·||S - T||²_F)))
    sum_sq = 0.0
    for i in range(n):
        xi = X[i:i+1].T  # (q,1)
        diff = xi @ xi.T - S
        sum_sq += np.sum(diff ** 2)

    # 归一化
    phi = sum_sq / (n ** 2)  # 平均 Frobenius 范数平方
    gamma = np.sum((S - T) ** 2)  # 目标距离

    if gamma > 0:
        lambda_star = min(1.0, max(0.0, phi / gamma))
    else:
        lambda_star = 0.0

    # 收缩
    C_shrunk = (1.0 - lambda_star) * S + lambda_star * T

    return C_shrunk, lambda_star


def ledoit_wolf_shrinkage_fast(X: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Ledoit-Wolf 收缩的向量化快速实现 (使用 sklearn).

    参数:
        X: (n_samples, n_features) 数据中心化后的矩阵

    返回:
        C_shrunk: (n_features, n_features) 收缩后的协方差矩阵
        lambda_star: 最优收缩强度 λ*
    """
    try:
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf().fit(X)
        return lw.covariance_, lw.shrinkage_
    except ImportError:
        warnings.warn("sklearn not available, using slow Ledoit-Wolf implementation")
        return ledoit_wolf_shrinkage(X)


def compute_intrinsic_metric(positions: np.ndarray,
                              align_reference: str = 'first',
                              shrinkage: str = 'ledoit_wolf',
                              eps_fallback: float = 0.01) -> Dict:
    """
    完整管线: Kabsch对齐 → 内禀协方差 → 收缩 → g_S (Law 1 v2 核心)

    参数:
        positions: (n_samples, n_residues, 3) 或 (n_samples, n_residues*3) 构象坐标
        align_reference: 对齐参考模式 ('first', 'mean', 'medoid')
        shrinkage: 收缩方法 ('ledoit_wolf', 'fixed', 'none')
        eps_fallback: 固定收缩时的 ε 值 (仅当 shrinkage='fixed')

    返回:
        字典, 包含:
            'aligned': (n_samples, n_residues, 3) 对齐后构象
            'centered': (n_samples, n_features) 中心化坐标
            'cov': (n_features, n_features) 协方差矩阵 (收缩后)
            'lambda_star': 收缩强度 (Ledoit-Wolf) 或固定 ε
            'g_S': (n_features, n_features) 精度矩阵 (正则化逆协方差)
            'eigenvalues': 特征值谱 (降序)
            'n_zero_eigs': 近零特征值个数 (应≈6, 对应刚体模式)
            'mean_pos': 平均构象 (3N维)
    """
    # 维度处理
    if positions.ndim == 3:
        n_samples, n_residues, _ = positions.shape
        pos_3d = positions
    else:
        n_samples = positions.shape[0]
        n_residues = positions.shape[1] // 3
        pos_3d = positions.reshape(n_samples, n_residues, 3)

    # Step 1: Kabsch 对齐 (去除 SE(3) 刚体自由度)
    aligned_3d = kabsch_align_ensemble(pos_3d, reference_mode=align_reference)

    # Step 2: 展平并中心化
    X_flat = aligned_3d.reshape(n_samples, n_residues * 3)
    mean_pos = X_flat.mean(axis=0)
    centered = X_flat - mean_pos

    # Step 3: 协方差估计 (收缩)
    q = centered.shape[1]
    if shrinkage == 'ledoit_wolf':
        cov, lambda_star = ledoit_wolf_shrinkage_fast(centered)
    elif shrinkage == 'fixed':
        S = (centered.T @ centered) / (n_samples - 1)
        eps = eps_fallback * np.trace(S) / q
        cov = S + eps * np.eye(q)
        lambda_star = eps_fallback
    elif shrinkage == 'none':
        cov = (centered.T @ centered) / (n_samples - 1)
        lambda_star = 0.0
    else:
        raise ValueError(f"Unknown shrinkage: {shrinkage}")

    # Step 4: 精度矩阵 g_S = (C + λI)^{-1}
    # 使用 Cholesky 分解或特征值截断保证数值稳定
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.maximum(eigenvalues, 1e-12)  # 数值截断

    # 构建 g_S (伪逆, 但已收缩故条件数可控)
    g_S = eigenvectors @ np.diag(1.0 / eigenvalues) @ eigenvectors.T

    # Step 5: 验证 — 近零特征值个数 (应≈6, 对应刚体模式)
    # 注意: 收缩后刚体模式不再是精确零, 但应远小于其他特征值
    threshold = eigenvalues.max() * 1e-6
    n_zero_eigs = int(np.sum(eigenvalues < threshold))

    return {
        'aligned': aligned_3d,
        'centered': centered,
        'cov': cov,
        'lambda_star': float(lambda_star),
        'g_S': g_S,
        'eigenvalues': eigenvalues[::-1],  # 降序
        'n_zero_eigs': n_zero_eigs,
        'mean_pos': mean_pos,
        'n_samples': n_samples,
        'n_residues': n_residues,
        'n_features': q,
        'shrinkage': shrinkage,
        'align_reference': align_reference,
    }


def compute_cgeo_intrinsic(z: np.ndarray, g_S: np.ndarray) -> float:
    """
    计算内禀几何扰动代价 C_geo = z^T g_S z (Law 1 v2)

    参数:
        z: (n_features,) 扰动向量 (构象变化)
        g_S: (n_features, n_features) 精度矩阵 (来自 compute_intrinsic_metric)

    返回:
        C_geo: 几何扰动代价 (正则化 Mahalanobis 距离)
    """
    return float(z @ g_S @ z)


# ============================================================
# 便捷函数: 从 NPZ 直接计算
# ============================================================

def load_and_compute_intrinsic(npz_dir: str,
                                align_reference: str = 'first',
                                shrinkage: str = 'ledoit_wolf',
                                max_samples: Optional[int] = None) -> Dict:
    """
    从 BioEmu NPZ 目录直接加载并计算内禀度量.

    参数:
        npz_dir: BioEmu 输出目录 (含 batch_*.npz)
        align_reference: 对齐参考模式
        shrinkage: 收缩方法
        max_samples: 最大采样数 (None=全部)

    返回:
        compute_intrinsic_metric 的结果字典
    """
    import pathlib
    npz_dir = pathlib.Path(npz_dir)
    npz_files = sorted(npz_dir.glob("batch_*.npz"))

    all_pos = []
    for f in npz_files:
        data = np.load(f, allow_pickle=True)
        if 'pos' in data:
            all_pos.append(data['pos'])
        elif 'positions' in data:
            all_pos.append(data['positions'])

    if not all_pos:
        raise ValueError(f"No position data found in {npz_dir}")

    positions = np.concatenate(all_pos, axis=0)
    if max_samples and len(positions) > max_samples:
        positions = positions[:max_samples]

    return compute_intrinsic_metric(positions, align_reference, shrinkage)


if __name__ == '__main__':
    # 测试示例
    print("Phase L1: Kabsch Alignment + Intrinsic Metric")
    print("=" * 50)
    print("模块已加载. 使用 compute_intrinsic_metric() 或 load_and_compute_intrinsic()")
