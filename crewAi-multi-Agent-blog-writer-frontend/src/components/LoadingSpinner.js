import React from "react";
import styled, { keyframes } from "styled-components";

// 애니메이션 키프레임 정의
const spin = keyframes`
  0%, 100% {
    opacity: 1;
    transform: scale(1);
  }
  50% {
    opacity: 0.5;
    transform: scale(1.5);
  }
`;

const SpinnerContainer = styled.div`
	display: flex;
	flex-direction: column;
	justify-content: center;
	align-items: center;
	height: 100vh;
	position: fixed;
	top: 0;
	left: 0;
	width: 100%;
	background-color: rgba(255, 255, 255, 0.85);
	z-index: 9999;
`;

const SpinnerWrapper = styled.div`
	display: flex;
	justify-content: center;
	align-items: center;
`;

const Spinner = styled.div`
	width: 30px;
	height: 30px;
	margin: 0 5px; /* 수평 간격 설정 */
	background-color: #007bff;
	border-radius: 50%;
	animation: ${spin} 1.2s ease-in-out infinite;
`;

const LoadingText = styled.p`
	margin-top: 20px;
	font-size: 20px;
	color: #007bff;
	font-weight: 600;
	text-align: center;
	line-height: 1.5;
`;

const LoadingSpinner = () => {
	return (
		<SpinnerContainer>
			<SpinnerWrapper>
				<Spinner />
				<Spinner />
				<Spinner />
			</SpinnerWrapper>
			<LoadingText>
				잠시만 기다려 주세요
				<span style={{ marginLeft: "5px" }}>...</span> {/* "..." 텍스트 추가 */}
			</LoadingText>
		</SpinnerContainer>
	);
};

export default LoadingSpinner;
