# Math Pipeline — What / Why / How matrices change

This document maps each linear-algebra step in the renderer to the matrices it
uses, why they appear, and how they transform the data.

---

## End-to-end map

| # | Idea | Where | What the matrix *is* | How it *changes* the data |
|---|------|-------|----------------------|---------------------------|
| 1 | Matrix representation | Image buffer, mesh loader | Image: \(H\times W\) per RGB; Mesh: \(3\times V\) vertices | Raw pixels/vertices become operable matrices |
| 2 | Solve linear systems | Ray–triangle (Möller–Trumbore), ray–quadric | \(3\times 3\) hit system / quadratic | Hit \(t\), barycentrics, or miss |
| 3 | Rank | SVD spectrum of image | Singular values \(\sigma_1\ge\ldots\ge\sigma_r\) | Rank ≈ how many patterns matter |
| 4 | Linear independence | Truncated SVD | Keep first \(r\) singular vectors | Drop redundant noisy patterns |
| 5 | Orthogonalization | Camera Gram–Schmidt; Duff bounce frame | \(Q\in SO(3)\), \(Q^\top Q=I\), \(\det Q=+1\) | Skewed directions → orthonormal frames |
| 6 | Projection | Camera ray; shading / shadows | Camera basis; \(n\cdot \ell\) terms | 3D → ray / pixel / cosine shading |
| 7 | Least squares | Truncated SVD = optimal rank-\(r\) approx | \(\min\|A-A_r\|_F\) | Best denoised image of given rank |
| 8 | Eigenvalues | Ellipsoid \(A=R\,\mathrm{diag}(1/r_i^2)\,R^\top\) | Symmetric \(3\times 3\) \(A\) | Axes = eigenvectors; radii \(=1/\sqrt{\lambda}\) |
| 9 | Diagonalization | Eigen of \(A\); SVD of image | \(A=Q\Lambda Q^\top\), \(A=U\Sigma V^\top\) | Geometry/noise → simple spectra |
| 10 | Final output | `04_denoised.png` | Denoised \(H\times W\) channels | Noise reduction via low-rank rebuild |

---

## Stage-by-stage

### 1. Matrix representation
- **What:** Images and meshes are matrices.
- **Why:** Same type of object for SVD, \(Q^\top Q\) checks, eigen, transforms.
- **Representation:**
  - Image channel \(A\in\mathbb{R}^{H\times W}\) (e.g. \(450\times 800\)).
  - Mesh vertices \(V\in\mathbb{R}^{3\times V}\) (each **column** one vertex).
  - Poses as \(4\times 4\) homogeneous transforms.
- **Change:** Scene/mesh files → dense numeric matrices.

### 2. Solving \(Ax=b\) / quadratics
- **What:** Linear (and quadratic) solves for intersections.
- **Why:** Decide where a ray hits a triangle or quadric.
- **Representation:** Möller–Trumbore
  \[
  \begin{bmatrix} e_1 & e_2 & -d \end{bmatrix}
  \begin{bmatrix} \beta \\ \gamma \\ t \end{bmatrix}
  = o - v_0.
  \]
  Ellipsoid: map ray by \(M^{-1}\), solve unit-sphere quadratic.
- **Change:** Ray \((o,d)\) → hit \((t,p,n)\) or miss.
- **Code:** `core/triangle.cpp`, `core/ellipsoid.cpp`.

### 3–4. Rank and independence (SVD)
- **What:** \(A = U\Sigma V^\top\).
- **Why:** Monte Carlo noise is high-frequency / low-energy; structure lives in large \(\sigma_i\).
- **Change:** Keep \(\sigma_1,\ldots,\sigma_r\) (`svd_rank` in `config.json`); rebuild \(A_r\).

### 5. Orthogonalization
- **What:** Orthonormal bases.
- **Why:**
  1. **Camera:** `eye` / `look_at` / `up` are not perpendicular → \(Q=[\mathrm{right}\mid\mathrm{up}\mid\mathrm{back}]\).
  2. **Diffuse bounce:** Duff frame \(Q=[T\mid B\mid N]\); \(v_w = Q v_\ell\).
- **Checks:** \(Q^\top Q = I\), \(\det Q = +1\).
- **Code:** `stages/stage_basis.cpp`. Numbers: `docs/SINGLE_PIXEL_TRACE.json`.

### 6. Projection
- **What:** Pixel → ray; Lambert cosine \(n\cdot\ell\); shadow tests along light direction.
- **Why:** Image formation and shading are projections onto frames / directions.
- **Change:** 3D scene ↔ 2D pixel grid + shading weights.

### 7. Least-squares low-rank fit
- **What:** Truncated SVD.
- **Why:** Unique best rank-\(r\) matrix in Frobenius norm (Eckart–Young–Mirsky).
- **Change:** Noisy path-traced \(A\) → denoised \(A_r\).
- **Show:** `03_pathtraced.png` → `04_denoised.png`.

### 8–9. Eigenvalues and diagonalization
- **What:** Symmetric eigen-decomposition; SVD related by \(\sigma_i=\sqrt{\lambda_i(A^\top A)}\).
- **Why:** Ellipsoid \(x^\top A x = 1\). Eigenvectors = axes; \(\lambda_i = 1/r_i^2\).
- **Representation:** \(A = R\,\mathrm{diag}(1/r_x^2,1/r_y^2,1/r_z^2)\,R^\top\).
- **Also:** PCA boxes in prep (`docs/prep_stage_trace.json`, stage `eigen`).

### 10. Final output
- Denoised HDR→PNG render after path tracing + SVD.

---

## Path tracing vs direct (quick visual check)

| Cue | Direct (`02_direct`) | Path traced (`03_pathtraced`) |
|-----|----------------------|--------------------------------|
| Shadows | Hard | Soft / noisy penumbra |
| Occluded faces | Nearly black | Bounce fill |
| Glass | Specular only | Soft caustic on floor |
| Noise | Low | Monte Carlo grain → SVD |

---

## Related files

| File | Contents |
|------|----------|
| `docs/MATH_PIPELINE.md` | This map |
| `docs/SINGLE_PIXEL_TRACE.json` | One pixel: camera \(Q\), ray, hit, bounce \(Q\), ellipsoid \(A\), toy SVD |
| `docs/prep_stage_trace.json` | Machine prep trace (`mesh_checks` + `eigen`) |
| `docs/TRACE_PLAN.md` | Optional live path-bounce logger plan |
