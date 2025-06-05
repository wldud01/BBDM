from inspect import isfunction


def extract(a, t, x_shape):
    """
    a: [N] 형태의 배열 (예: betas, alphas 등)
    t: [B] 형태의 timestep index tensor
    x_shape: 결과 텐서의 shape 기준 (e.g., [B, C, H, W])
    
    -> 각 배치마다 timestep에 해당하는 값 a[t]를 추출하고,
       broadcasting을 위해 [B, 1, 1, 1] 등으로 reshape
    """
    b, *_ = t.shape
    out = a.gather(-1, t)   # 각 배치별 timestep 위치에서 값 추출
    return out.reshape(b, *((1,) * (len(x_shape) - 1))) # shape 맞춰서 broadcast 준비


def exists(x):
    """None이 아닌 값인지 확인"""
    return x is not None


def default(val, d):
    """
    val이 존재하면 그대로 반환하고, 없으면 기본값 d 사용
    d는 값 또는 함수일 수 있음
    """
    if exists(val):
        return val
    return d() if isfunction(d) else d
