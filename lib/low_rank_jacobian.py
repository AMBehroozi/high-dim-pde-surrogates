import torch
import torch.nn.functional as F
import torch.nn as nn


def compute_low_rank_jacobian_3(model, y, x, rank=5, epsilon=1e-4, seed=None):
    """
    Compute a fully differentiable low-rank approximation of the Jacobian
    by directly using the model for perturbations.
    """
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        
    # Get dimensions
    nb, nx, ny = x.shape
    
    # Extract the last time step of the original output
    y_last = y[..., -1]
    
    # Generate random probing directions
    v_probes = torch.randn(rank, nb, nx, ny, device=x.device)
    v_probes = v_probes / torch.norm(v_probes.reshape(rank, -1), dim=1).reshape(rank, 1, 1, 1)
    
    
    # Initialize U tensor to store responses
    u_responses = []
    
    # Probe the model with each direction
    for i in range(rank):
        # Add small perturbation to input
        x_perturbed = x + epsilon * v_probes[i]
        
        # Forward pass with perturbed input
        y_perturbed = model(x_perturbed)[..., -1]
        
        # Compute response (difference in output divided by epsilon)
        u_i = (y_perturbed - y_last) / epsilon
        u_responses.append(u_i)
    
    # Stack responses along a new dimension
    U = torch.stack(u_responses, dim=-1)  # [nb, nx, ny, rank]
    V = v_probes.permute(1, 2, 3, 0)  # [nb, nx, ny, rank]
    
    # Reshape to matrix form
    U_mat = U.reshape(nb, nx*ny, rank)
    V_mat = V.reshape(nb, nx*ny, rank)
    
    return U_mat, V_mat
    


def compute_low_rank_jacobian_1(model, y, x, u, rank=5, epsilon=1e-4, seed=None):
    """
    Compute a fully differentiable low-rank approximation of the Jacobian
    by directly using the model for perturbations.
    """
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        
    # Get dimensions
    nb, nx, ny = x.shape
    
    # Extract the last time step of the original output
    y_last = y[..., -1]
    
    # Generate random probing directions
    v_probes = torch.randn(rank, nb, nx, ny, device=x.device)
    v_probes = v_probes / torch.norm(v_probes.reshape(rank, -1), dim=1).reshape(rank, 1, 1, 1)
    
    
    # Initialize U tensor to store responses
    u_responses = []
    
    # Probe the model with each direction
    for i in range(rank):
        # Add small perturbation to input
        x_perturbed = x + epsilon * v_probes[i]
        
        # Forward pass with perturbed input
        y_perturbed = model(u, x_perturbed)[..., -1]
        
        # Compute response (difference in output divided by epsilon)
        u_i = (y_perturbed - y_last) / epsilon
        u_responses.append(u_i)
    
    # Stack responses along a new dimension
    U = torch.stack(u_responses, dim=-1)  # [nb, nx, ny, rank]
    V = v_probes.permute(1, 2, 3, 0)  # [nb, nx, ny, rank]
    
    # Reshape to matrix form
    U_mat = U.reshape(nb, nx*ny, rank)
    V_mat = V.reshape(nb, nx*ny, rank)
    
    return U_mat, V_mat
    
def compute_low_rank_jacobian_2(model, y, x, u, rank=5, epsilon=1e-4, seed=None):
    """
    Compute a differentiable low-rank approximation of the Jacobian of y = model(u, x)[..., -1]
    with respect to x.
    
    Args:
        model (nn.Module): Neural network that takes (u, x) and returns a tensor with last dimension T
        u (torch.Tensor): Input tensor of shape [nb, nx, ny, T_in]
        x (torch.Tensor): Input tensor of shape [nb, nx, ny]
        rank (int): Rank of the approximation (number of probing directions)
        epsilon (float): Small perturbation size for probing
        seed (int, optional): Random seed for reproducibility
        
    Returns:
        tuple: (U, V) tensors for low-rank approximation, each of shape [nb, nx*ny, rank]
               jacobian_op: A function that applies the approximated Jacobian to vectors
    """
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        
    # Get dimensions
    nb, nx, ny = x.shape
    
    # Ensure x requires gradient
    x_requires_grad = x.requires_grad
    if not x_requires_grad:
        x = x.detach().requires_grad_(True)
    
    # Compute base output
    with torch.set_grad_enabled(True):
        y_base = y[..., -1]
    
    # Generate random probing directions
    v_probes = torch.randn(rank, nb, nx, ny, device=x.device)
    # Normalize each probing direction
    v_probes = v_probes / torch.norm(v_probes.reshape(rank, -1), dim=1).reshape(rank, 1, 1, 1)
    
    # Initialize U tensor to store responses
    u_responses = []
    
    # Probe the model with each direction
    for i in range(rank):
        # Add small perturbation to input
        x_perturbed = x + epsilon * v_probes[i]
        
        # Forward pass with perturbed input
        with torch.set_grad_enabled(True):
            y_perturbed = model(u, x_perturbed)[..., -1]
        
        # Compute response (difference in output divided by epsilon)
        u_i = (y_perturbed - y_base) / epsilon
        u_responses.append(u_i)
    
    # Stack responses along a new dimension
    U = torch.stack(u_responses, dim=-1)  # [nb, nx, ny, rank]
    V = v_probes.permute(1, 2, 3, 0)  # [nb, nx, ny, rank]
    
    # Reshape to matrix form
    U_mat = U.reshape(nb, nx*ny, rank)
    V_mat = V.reshape(nb, nx*ny, rank)
    
    if not x_requires_grad:
        x = x.detach()
    
    return U_mat, V_mat


def sliced_jacobian_to_low_rank(sliced_jacobian, full_nx, full_ny, rank=5, interpolation_mode='bilinear'):
    """
    Convert a sliced/subsampled Jacobian tensor to a low-rank approximation of the full Jacobian.
    
    Args:
        sliced_jacobian (torch.Tensor): Subsampled Jacobian tensor of shape [nb, nx_small, ny_small, nx_small, ny_small]
        full_nx (int): The full x-dimension size
        full_ny (int): The full y-dimension size
        rank (int): Rank of the approximation
        interpolation_mode (str): Mode for upscaling ('nearest', 'bilinear', 'bicubic')
        
    Returns:
        tuple: (U, V) tensors for low-rank approximation, each of shape [nb, full_nx*full_ny, rank]
               jacobian_op: A function that applies the approximated full Jacobian to vectors
    """
    # Get dimensions
    nb, nx_small, ny_small, nx_small2, ny_small2 = sliced_jacobian.shape
    assert nx_small == nx_small2 and ny_small == ny_small2, "Spatial dimensions must match"
    
    # Verify that dimensions are properly related (divisible)
    assert full_nx % nx_small == 0 and full_ny % ny_small == 0, "Full dimensions should be multiples of slice dimensions"
    
    # Get device from input tensor
    device = sliced_jacobian.device
    
    # Compute scale factors
    scale_x = full_nx // nx_small
    scale_y = full_ny // ny_small
    
    # Reshape to batch of matrices for SVD
    J_slice_flat = sliced_jacobian.reshape(nb, nx_small*ny_small, nx_small*ny_small)
    
    # Initialize U and V tensors for the sliced Jacobian
    U_small = torch.zeros(nb, nx_small*ny_small, rank, device=device)
    V_small = torch.zeros(nb, nx_small*ny_small, rank, device=device)
    
    # Process each batch with SVD
    for b in range(nb):
        # Compute SVD
        try:
            u, s, v = torch.svd(J_slice_flat[b])
        except Exception:
            print(f"Warning: SVD failed for batch {b}, using more stable approach")
            u, s, v = torch.svd_lowrank(J_slice_flat[b], q=min(nx_small*ny_small, rank*2))
        
        # Truncate to rank r
        u_r = u[:, :rank]
        s_r = s[:rank]
        v_r = v[:, :rank]
        
        # Absorb singular values
        sqrt_s = torch.sqrt(s_r)
        U_small[b] = u_r * sqrt_s.unsqueeze(0)
        V_small[b] = v_r * sqrt_s.unsqueeze(0)
    
    # Reshape U_small and V_small to 4D tensors for spatial interpolation
    U_small_spatial = U_small.reshape(nb, nx_small, ny_small, rank)
    V_small_spatial = V_small.reshape(nb, nx_small, ny_small, rank)
    
    # Transpose to [nb, rank, nx_small, ny_small] for interpolation
    U_small_spatial = U_small_spatial.permute(0, 3, 1, 2)
    V_small_spatial = V_small_spatial.permute(0, 3, 1, 2)
    
    # Upscale to full resolution using interpolation
    U_full_spatial = F.interpolate(
        U_small_spatial, 
        size=(full_nx, full_ny), 
        mode=interpolation_mode, 
        align_corners=False if interpolation_mode != 'nearest' else None
    )
    
    V_full_spatial = F.interpolate(
        V_small_spatial, 
        size=(full_nx, full_ny), 
        mode=interpolation_mode, 
        align_corners=False if interpolation_mode != 'nearest' else None
    )
    
    # Transpose back to [nb, full_nx, full_ny, rank]
    U_full_spatial = U_full_spatial.permute(0, 2, 3, 1)
    V_full_spatial = V_full_spatial.permute(0, 2, 3, 1)
    
    # Reshape to final U and V matrices
    U = U_full_spatial.reshape(nb, full_nx*full_ny, rank)
    V = V_full_spatial.reshape(nb, full_nx*full_ny, rank)
        
    return torch.stack((U, V), dim=-1)




def compute_low_rank_jacobian_loss(du_dp__low_rank_true, U_pred, V_pred, method='action', random_seed=42):
    
    torch.manual_seed(42) 
    U_true = du_dp__low_rank_true[..., 0]
    V_true = du_dp__low_rank_true[..., 1]
    
    """
    Compute loss between true and predicted low-rank Jacobian approximations.
    
    Args:
        U_true (torch.Tensor): U matrix of true Jacobian, shape [nb, nx*ny, rank_true]
        V_true (torch.Tensor): V matrix of true Jacobian, shape [nb, nx*ny, rank_true]
        U_pred (torch.Tensor): U matrix of predicted Jacobian, shape [nb, nx*ny, rank_pred]
        V_pred (torch.Tensor): V matrix of predicted Jacobian, shape [nb, nx*ny, rank_pred]
        method (str): Loss method, one of 'frobenius', 'spectral', 'action', 'subspace', 'alignment'
        
    Returns:
        torch.Tensor: Loss value
    """
    nb = U_true.shape[0]
    
    if method == 'frobenius':
        # Frobenius norm of difference between full matrices
        # ||U_true @ V_true.T - U_pred @ V_pred.T||_F^2
        # This uses an efficient computation without materializing full matrices
        
        # Compute ||U_true @ V_true.T||_F^2
        UV_true_norm_sq = torch.sum(U_true**2) * torch.sum(V_true**2)
        
        # Compute ||U_pred @ V_pred.T||_F^2
        UV_pred_norm_sq = torch.sum(U_pred**2) * torch.sum(V_pred**2)
        
        # Compute <U_true @ V_true.T, U_pred @ V_pred.T>_F
        # = Tr(V_true @ U_true.T @ U_pred @ V_pred.T)
        # = sum_ij (U_true.T @ U_pred)_ij * (V_true.T @ V_pred)_ij
        U_similarity = torch.bmm(U_true.transpose(1, 2), U_pred)  # [nb, rank_true, rank_pred]
        V_similarity = torch.bmm(V_true.transpose(1, 2), V_pred)  # [nb, rank_true, rank_pred]
        cross_term = torch.sum(U_similarity * V_similarity)
        
        # ||A - B||_F^2 = ||A||_F^2 + ||B||_F^2 - 2<A,B>_F
        loss = UV_true_norm_sq + UV_pred_norm_sq - 2 * cross_term
        
        # Normalize by batch size and dimensionality for better scaling
        loss = loss / nb
        
    elif method == 'spectral':
        # Approximate spectral norm (maximum singular value of difference)
        # We use power iteration to estimate this
        
        # Function to apply (U_true @ V_true.T - U_pred @ V_pred.T) to a vector
        def apply_diff(vec):
            # Apply true Jacobian: U_true @ (V_true.T @ vec)
            V_true_T_vec = torch.bmm(V_true.transpose(1, 2), vec.unsqueeze(-1))
            true_result = torch.bmm(U_true, V_true_T_vec).squeeze(-1)
            
            # Apply pred Jacobian: U_pred @ (V_pred.T @ vec)
            V_pred_T_vec = torch.bmm(V_pred.transpose(1, 2), vec.unsqueeze(-1))
            pred_result = torch.bmm(U_pred, V_pred_T_vec).squeeze(-1)
            
            # Return difference
            return true_result - pred_result
        
        # Power iteration to estimate largest singular value
        dim = U_true.shape[1]  # nx*ny
        v = torch.randn(nb, dim, device=U_true.device)
        v = v / torch.norm(v, dim=1, keepdim=True)
        
        for _ in range(10):  # 10 iterations of power method
            # Apply J.T @ J @ v
            Jv = apply_diff(v)
            JTJv = apply_diff(Jv)
            
            # Normalize
            v = JTJv / torch.norm(JTJv, dim=1, keepdim=True)
        
        # Estimate spectral norm
        Jv = apply_diff(v)
        loss = torch.mean(torch.norm(Jv, dim=1))

    elif method == 'action':
        # Set seed for reproducibility
        torch.manual_seed(42)
        
        num_vectors = 20  # Number of test vectors
        
        # Generate all test vectors at once for efficiency
        all_vectors = torch.randn(num_vectors, nb, U_true.shape[1], device=U_true.device)
        
        # Initialize tensors to store results
        all_true_results = []
        all_pred_results = []
        
        # Process each test vector
        for i in range(num_vectors):
            vec = all_vectors[i]
            
            # Apply true Jacobian
            V_true_T_vec = torch.bmm(V_true.transpose(1, 2), vec.unsqueeze(-1))
            true_result = torch.bmm(U_true, V_true_T_vec).squeeze(-1)
            
            # Apply pred Jacobian
            V_pred_T_vec = torch.bmm(V_pred.transpose(1, 2), vec.unsqueeze(-1))
            pred_result = torch.bmm(U_pred, V_pred_T_vec).squeeze(-1)
            
            all_true_results.append(true_result)
            all_pred_results.append(pred_result)
        
        # Stack results
        true_stack = torch.stack(all_true_results)
        pred_stack = torch.stack(all_pred_results)
        
        # Compute MSE over all results
        loss = F.mse_loss(true_stack, pred_stack)

    # elif method == 'action':
    #     # Set seed for reproducibility
    #     torch.manual_seed(42)
        
    #     num_vectors = 20  # Increase number of test vectors
    #     total_error = 0.0
        
    #     for _ in range(num_vectors):
    #         # Generate random vector (without normalization)
    #         vec = torch.randn(nb, U_true.shape[1], device=U_true.device)
            
    #         # Apply true Jacobian: U_true @ (V_true.T @ vec)
    #         V_true_T_vec = torch.bmm(V_true.transpose(1, 2), vec.unsqueeze(-1))
    #         true_result = torch.bmm(U_true, V_true_T_vec).squeeze(-1)
            
    #         # Apply pred Jacobian: U_pred @ (V_pred.T @ vec)
    #         V_pred_T_vec = torch.bmm(V_pred.transpose(1, 2), vec.unsqueeze(-1))
    #         pred_result = torch.bmm(U_pred, V_pred_T_vec).squeeze(-1)
            
    #         # Compute relative error using L1 norm for more sensitivity
    #         # This will be more sensitive to differences in magnitude
    #         error = torch.abs(true_result - pred_result).sum() / (torch.abs(true_result).sum() + 1e-6)
    #         total_error += error
        
    #     loss = total_error / num_vectors
        
    elif method == 'subspace':
        # Compare the subspaces spanned by the columns of U and V
        # We use principal angles between subspaces
        
        rank_true = U_true.shape[2]
        rank_pred = U_pred.shape[2]
        
        # Orthogonalize the columns of U_true and U_pred
        U_true_ortho = torch.zeros_like(U_true)
        U_pred_ortho = torch.zeros_like(U_pred)
        
        for b in range(nb):
            # QR decomposition for orthogonalization
            q_true, _ = torch.linalg.qr(U_true[b])
            q_pred, _ = torch.linalg.qr(U_pred[b])
            
            # Store orthogonalized matrices
            U_true_ortho[b, :, :rank_true] = q_true[:, :rank_true]
            U_pred_ortho[b, :, :rank_pred] = q_pred[:, :rank_pred]
        
        # Compute cosines of principal angles (singular values of U_true_ortho.T @ U_pred_ortho)
        cosines = torch.bmm(U_true_ortho.transpose(1, 2), U_pred_ortho)
        
        # Compute loss as sum of squared sines of angles
        # sin²(θ) = 1 - cos²(θ)
        loss = torch.mean(torch.sum(1 - torch.pow(cosines, 2), dim=(1, 2)))
        
    elif method == 'alignment':
        # Measure alignment of the low-rank factors
        # This is helpful when the true and predicted ranks are the same
        
        if U_true.shape[2] != U_pred.shape[2]:
            raise ValueError("For alignment loss, true and predicted ranks must be the same")
        
        rank = U_true.shape[2]
        
        # We need to find the best alignment of the factors
        # Since each column pair (u_i, v_i) can be scaled by α and 1/α without changing the product
        
        # Compute all pairwise alignments between columns
        alignment_scores = torch.zeros(nb, rank, rank, device=U_true.device)
        
        for i in range(rank):
            u_true_i = U_true[:, :, i].unsqueeze(-1)  # [nb, nx*ny, 1]
            v_true_i = V_true[:, :, i].unsqueeze(-1)  # [nb, nx*ny, 1]
            
            for j in range(rank):
                u_pred_j = U_pred[:, :, j].unsqueeze(-1)  # [nb, nx*ny, 1]
                v_pred_j = V_pred[:, :, j].unsqueeze(-1)  # [nb, nx*ny, 1]
                
                # Compute cosine similarity between u vectors
                u_sim = torch.bmm(u_true_i.transpose(1, 2), u_pred_j).squeeze()
                u_sim = u_sim / (torch.norm(u_true_i, dim=1) * torch.norm(u_pred_j, dim=1)).squeeze()
                
                # Compute cosine similarity between v vectors
                v_sim = torch.bmm(v_true_i.transpose(1, 2), v_pred_j).squeeze()
                v_sim = v_sim / (torch.norm(v_true_i, dim=1) * torch.norm(v_pred_j, dim=1)).squeeze()
                
                # Combined alignment score
                alignment_scores[:, i, j] = (u_sim.abs() * v_sim.abs()).squeeze()
        
        # Use a greedy approach for matching columns
        loss = 0.0
        for b in range(nb):
            scores = alignment_scores[b].clone()  # [rank, rank]
            matched_score = 0.0
            remaining_rows = list(range(rank))
            remaining_cols = list(range(rank))
            
            # Greedy matching
            while len(remaining_rows) > 0 and len(remaining_cols) > 0:
                # Get current submatrix of scores
                curr_scores = scores[remaining_rows, :][:, remaining_cols]
                
                # Find max score
                max_val, max_idx = torch.max(curr_scores.view(-1), dim=0)
                local_i = max_idx.item() // len(remaining_cols)
                local_j = max_idx.item() % len(remaining_cols)
                
                # Map to original indices
                i = remaining_rows[local_i]
                j = remaining_cols[local_j]
                
                # Add to matched score
                matched_score += scores[i, j]
                
                # Remove the assigned row and column
                remaining_rows.remove(i)
                remaining_cols.remove(j)
            
            # Compute loss for this batch (1 - average alignment)
            loss += 1.0 - (matched_score / rank)
        
        loss = loss / nb
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return loss

