import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from tqdm.autonotebook import tqdm
import numpy as np

from model.utils import extract, default
from model.BrownianBridge.base.modules.diffusionmodules.openaimodel import UNetModel
#from model.BrownianBridge.base.modules.encoders.modules import SpatialRescaler


class BrownianBridgeModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config

        # 모델 하이퍼파라미터 초기화 및 schedule 설정
        model_params = model_config.BB.params  
        self.num_timesteps = model_params.num_timesteps  
        self.mt_type = model_params.mt_type 
        self.max_var = model_params.max_var if model_params.__contains__("max_var") else 1  
        self.eta = model_params.eta if model_params.__contains__("eta") else 1  
        self.skip_sample = model_params.skip_sample  #
        self.sample_type = model_params.sample_type  
        self.sample_step = model_params.sample_step  
        self.steps = None 
        self.register_schedule()  

        # 학습 설정 (loss function 및 objective type 등)
        self.loss_type = model_params.loss_type  
        self.objective = model_params.objective  

        # UNet 기반의 denoising function 정의
        self.image_size = model_params.UNetParams.image_size  # 입력 이미지 크기
        self.channels = model_params.UNetParams.in_channels  # 입력 채널 수
        self.condition_key = model_params.UNetParams.condition_key  # 조건 방식 설정 (e.g., 'nocond')
        self.denoise_fn = UNetModel(**vars(model_params.UNetParams))  # denoising을 위한 UNet 모델 초기화



    def register_schedule(self):
        """
        Forward diffusion 과정을 위한 m_t, variance 등의 스케줄 정의
        - m_t: 시간에 따른 interpolation 계수 (x0과 y를 혼합하는 비율)
        - variance: Brownian Bridge에서의 시점별 노이즈 분산
        - posterior_variance: reverse 과정에서 사용할 분산
        """
        # 총 timestep 수
        T = self.num_timesteps

        # m_t 정의 (시간에 따른 interpolation weight)
        if self.mt_type == "linear":
            m_min, m_max = 0.001, 0.999
            m_t = np.linspace(m_min, m_max, T)
        elif self.mt_type == "sin":
            m_t = 1.0075 ** np.linspace(0, T, T)
            m_t = m_t / m_t[-1]
            m_t[-1] = 0.999 # 마지막 값은 1에 수렴하게 설정정
        else:
            raise NotImplementedError
        
        # 이전 timestep의 m_t (t-1 시점용)
        m_tminus = np.append(0, m_t[:-1])

        # variance 계산 (Brownian Bridge 이론 기반)
        variance_t = 2. * (m_t - m_t ** 2) * self.max_var   
        variance_tminus = np.append(0., variance_t[:-1])    
        
        # t 와 t-1 간의 분산 차이 (Bridge 기반 조정)
        variance_t_tminus = variance_t - variance_tminus * ((1. - m_t) / (1. - m_tminus)) ** 2

        # reverse step에서 사용하는 값
        posterior_variance_t = variance_t_tminus * variance_tminus / variance_t

        # torch tensor로 변환 후 register_buffer로 저장 (학습 중 유지됨)
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('m_t', to_torch(m_t))
        self.register_buffer('m_tminus', to_torch(m_tminus))
        self.register_buffer('variance_t', to_torch(variance_t))
        self.register_buffer('variance_tminus', to_torch(variance_tminus))
        self.register_buffer('variance_t_tminus', to_torch(variance_t_tminus))
        self.register_buffer('posterior_variance_t', to_torch(posterior_variance_t))

        # Reverse sampling 시 사용할 step 리스트 구성
        if self.skip_sample:
            # 예: 1000 step 중 일부만 선택해 sampling (linear 간격)
            if self.sample_type == 'linear':
                midsteps = torch.arange(self.num_timesteps - 1, 1,
                                        step=-((self.num_timesteps - 1) / (self.sample_step - 2))).long()
                self.steps = torch.cat((midsteps, torch.Tensor([1, 0]).long()), dim=0)
            elif self.sample_type == 'cosine':
                # cosine 스케줄 기반으로 sampling step 선택
                steps = np.linspace(start=0, stop=self.num_timesteps, num=self.sample_step + 1)
                steps = (np.cos(steps / self.num_timesteps * np.pi) + 1.) / 2. * self.num_timesteps
                self.steps = torch.from_numpy(steps)
        else:
            # sampling을 모두 사용하는 경우 (역순)
            self.steps = torch.arange(self.num_timesteps-1, -1, -1)

    def apply(self, weight_init):
        # UNet 파라미터에 초기화 함수 적용
        self.denoise_fn.apply(weight_init)
        return self

    def get_parameters(self):
        # UNet 파라미터 반환
        return self.denoise_fn.parameters()

    def forward(self, x, y, context=None):
        """
        학습 시 호출되는 entry point 함수
        - x: target 도메인 이미지 (예: CECT)
        - y: source 도메인 이미지 (예: NECT)
        - context: UNet의 ondition input으로 주입하는 정보 (cross-attention 등)
        """

        # 조건 없이 학습할 경우 context 제거
        if self.condition_key == "nocond":
            context = None
        else:
            # context가 명시되지 않으면 기본적으로 y를 조건으로 사용
            context = y if context is None else context

        # batch 크기, 채널, 높이, 너비, device 정보 추출
        b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # 이미지 크기 확인
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        # 각 배치마다 random으로 timestep을 샘플링 (0 ~ T-1)
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        
        # 손실 계산 및 x0 복원 수행
        return self.p_losses(x, y, context, t)

    def p_losses(self, x0, y, context, t, noise=None):
        """
        모델 학습 시 사용하는 주요 손실 함수

        Args:
            x0 (Tensor): 복원 대상 이미지 (target domain, e.g., CECT)
            y (Tensor): diffusion 종점이자 조건 (source domain, e.g., NECT)
            context (Tensor): UNet에 주어지는 조건 정보 (보통 y와 동일)
            t (Tensor): timestep (각 샘플마다 다를 수 있음)
            noise (Tensor, optional): 노이즈 입력. 주어지지 않으면 랜덤 생성됨

        Returns:
            recloss (Tensor): 예측된 objective와 정답 간의 재구성 손실
            log_dict (dict): 손실과 예측 결과(x0 복원본) 포함한 로그
        """

        b, c, h, w = x0.shape
        # 노이즈가 주어지지 않으면 표준 정규분포에서 생성
        noise = default(noise, lambda: torch.randn_like(x0))

        # Forward process: x₀를 노이즈화하여 xₜ 생성
        x_t, objective = self.q_sample(x0, y, t, noise)
        # Reverse process: xₜ로부터 objective 예측
        objective_recon = self.denoise_fn(x_t, timesteps=t, context=context)

        # loss type: 'l1' 또는 'l2' 중 선택
        if self.loss_type == 'l1':
            recloss = (objective - objective_recon).abs().mean()
        elif self.loss_type == 'l2':
            recloss = F.mse_loss(objective, objective_recon)
        else:
            raise NotImplementedError()
        
        # xₜ에서 x₀ 복원 (sampling 시 활용 가능) 
        x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon)
        
        # 로그 정보로 결과 구성
        log_dict = {
            "loss": recloss,
            "x0_recon": x0_recon
        }
        
        return recloss, log_dict

    def q_sample(self, x0, y, t, noise=None):
        """
        Forward diffusion: x₀ → xₜ

        BBDM의 공식에 따라 x₀에서 y로 이동하는 중간 상태 xₜ를 샘플링

        Args:
            x0 (Tensor): 복원 대상 이미지 (target)
            y (Tensor): source 이미지 (bridge의 종점)
            t (Tensor): 현재 timestep
            noise (Tensor, optional): 추가할 가우시안 노이즈

        Returns:
            x_t (Tensor): x₀에서 y로 이동 중인 경로의 한 시점
            objective (Tensor): 학습 시 예측 대상 (objective 종류에 따라 다름)
        """
        # x_0크기에 맞춰서 noise 생성
        noise = default(noise, lambda: torch.randn_like(x0))
        
        # 현재 timestep t에 해당하는 m_t와 분산 (var_t), 표준편차 (sigma_t) 추출
        m_t = extract(self.m_t, t, x0.shape)
        var_t = extract(self.variance_t, t, x0.shape)
        sigma_t = torch.sqrt(var_t)

        # objective 종류
        if self.objective == 'grad':   
            objective = m_t * (y - x0) + sigma_t * noise
        elif self.objective == 'noise':   
            objective = noise
        elif self.objective == 'ysubx':  
            objective = y - x0
        else:
            raise NotImplementedError()

        # BBDM forward equation: xₜ = (1-mₜ)x₀ + mₜy + σₜε
        return (
            (1. - m_t) * x0 + m_t * y + sigma_t * noise,
            objective
        )

    def predict_x0_from_objective(self, x_t, y, t, objective_recon):
        """
        예측된 objective를 사용하여 x₀ 복원
        (즉, 역방향 샘플링에서 초기 상태 x₀를 추정)

        Args:
            x_t (Tensor): 현재 시점의 중간 상태 (경로 위의 점)
            y (Tensor): source 도메인 이미지 (브릿지 종점)
            t (Tensor): timestep
            objective_recon (Tensor): 모델이 예측한 objective 값

        Returns:
            x0_recon (Tensor): 복원된 x₀
        """
        if self.objective == 'grad':
            x0_recon = x_t - objective_recon
        elif self.objective == 'noise':
            m_t = extract(self.m_t, t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            sigma_t = torch.sqrt(var_t)
            x0_recon = (x_t - m_t * y - sigma_t * objective_recon) / (1. - m_t)
        elif self.objective == 'ysubx':
            x0_recon = y - objective_recon
        else:
            raise NotImplementedError
        return x0_recon

    @torch.no_grad()
    def q_sample_loop(self, x0, y):
        """
        Forward 과정을 시각화용으로 step-by-step 수행
        - x0 → x1 → ... → xT
        """
        imgs = [x0]
        for i in tqdm(range(self.num_timesteps), desc='q sampling loop', total=self.num_timesteps):
            t = torch.full((y.shape[0],), i, device=x0.device, dtype=torch.long)
            img, _ = self.q_sample(x0, y, t)
            imgs.append(img)
        return imgs

    @torch.no_grad()
    def p_sample(self, x_t, y, context, i, clip_denoised=False):
        """
        Reverse sampling의 단일 step 수행 (x_t → x_{t-1})
        
        Args:
            x_t (Tensor): 현재 시점의 경로 상태 (noisy intermediate sample)
            y (Tensor): source 도메인 (브릿지의 종점)
            context (Tensor): UNet 조건 입력
            i (int): sampling step index
            clip_denoised (bool): 복원된 x₀ 클리핑 여부

        Returns:
            x_{t-1} 샘플, 복원된 x₀
        """
        b, *_, device = *x_t.shape, x_t.device
        
        # 마지막 step일 경우: x₀ 직접 출력
        if self.steps[i] == 0:
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            objective_recon = self.denoise_fn(x_t, timesteps=t, context=context)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)
            return x0_recon, x0_recon
        else:
            # 일반 step: x_{t-1} 샘플링 수행
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            n_t = torch.full((x_t.shape[0],), self.steps[i+1], device=x_t.device, dtype=torch.long)

            # objective 예측 및 x₀ 복원
            objective_recon = self.denoise_fn(x_t, timesteps=t, context=context)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)

            # Brownian Bridge 기반 posterior mean 및 sigma 계산
            m_t = extract(self.m_t, t, x_t.shape)
            m_nt = extract(self.m_t, n_t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            var_nt = extract(self.variance_t, n_t, x_t.shape)
            sigma2_t = (var_t - var_nt * (1. - m_t) ** 2 / (1. - m_nt) ** 2) * var_nt / var_t
            sigma_t = torch.sqrt(sigma2_t) * self.eta

            noise = torch.randn_like(x_t)

            # posterior 평균 계산: BBDM 수식 기반
            x_tminus_mean = (1. - m_nt) * x0_recon + m_nt * y + torch.sqrt((var_nt - sigma2_t) / var_t) * \
                            (x_t - (1. - m_t) * x0_recon - m_t * y)

            # x_{t-1} 샘플 생성
            return x_tminus_mean + sigma_t * noise, x0_recon

    @torch.no_grad()
    def p_sample_loop(self, y, context=None, clip_denoised=True, sample_mid_step=False):
        """
        전체 Reverse sampling loop 수행
            - source 이미지 y에서 시작하여 target 이미지 x0로 복원
            - self.steps에 따라 역방향 샘플링 반복 수행
            - (선택적으로) 중간 결과들도 함께 반환

        Args:
            y (Tensor): source 도메인 이미지 (e.g., NECT)
            context (Tensor): 조건 입력 (default는 y 자체)
            clip_denoised (bool): 복원된 x0를 [-1, 1] 범위로 클리핑할지 여부
            sample_mid_step (bool): 중간 timestep 이미지들을 저장할지 여부

        Returns:
            - sample_mid_step=True인 경우: 전체 trajectory와 각 step의 x0 복원값 리스트
            - sample_mid_step=False인 경우: 최종 복원된 x0만 반환
        """
        # 조건 설정
        if self.condition_key == "nocond":
            context = None
        else:
            context = y if context is None else context

        # 전체 timestep에 대해 반복 수행
        if sample_mid_step:
            imgs, one_step_imgs = [y], [] # 중간 x_t trajectory 저장용, 매 step의 복원된 x0 저장용
            for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
                img, x0_recon = self.p_sample(x_t=imgs[-1], y=y, context=context, i=i, clip_denoised=clip_denoised)
                imgs.append(img)
                one_step_imgs.append(x0_recon)
            return imgs, one_step_imgs
        else:
            img = y
            for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
                img, _ = self.p_sample(x_t=img, y=y, context=context, i=i, clip_denoised=clip_denoised)
            return img

    @torch.no_grad()
    def sample(self, y, context=None, clip_denoised=True, sample_mid_step=False):
        # 샘플 생성 함수
        return self.p_sample_loop(y, context, clip_denoised, sample_mid_step)