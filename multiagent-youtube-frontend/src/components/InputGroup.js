import React from "react";
import styled from "styled-components";

const InputGroupWrapper = styled.div`
	display: flex;
	justify-content: center;
	margin-bottom: 20px;
	flex-direction: column;
	align-items: center;
`;

const Input = styled.input`
	width: 80%;
	max-width: 500px;
	padding: 12px;
	border: 1px solid #99ccff; /* 연한 파란색 테두리 */
	border-radius: 8px;
	margin-bottom: 10px;
	font-size: 16px;
	transition: border-color 0.3s ease, box-shadow 0.3s ease;

	&:focus {
		border-color: #3399ff; /* 더 진한 파란색 */
		outline: none;
		box-shadow: 0 0 5px rgba(51, 153, 255, 0.5); /* 파란색 그림자 */
	}
`;

const Button = styled.button`
	padding: 12px 24px;
	background-color: #3399ff; /* 기본 파란색 */
	color: #fff;
	border: none;
	border-radius: 8px;
	cursor: pointer;
	font-size: 16px;
	font-weight: bold;
	transition: background-color 0.3s ease, transform 0.3s ease;

	&:hover {
		background-color: #267dcc; /* 더 어두운 파란색 */
		transform: translateY(-2px); /* 버튼이 약간 올라가는 효과 */
	}

	&:disabled {
		background-color: #b3d1ff; /* 비활성화된 상태의 연한 파란색 */
		cursor: not-allowed;
	}
`;

const InputGroup = ({ youtubeUrl, handleInputChange, fetchData, loading }) => {
	return (
		<InputGroupWrapper>
			<Input type="text" placeholder="YouTube URL을 입력하세요" value={youtubeUrl} onChange={handleInputChange} />
			<Button onClick={fetchData} disabled={loading || !youtubeUrl}>
				{loading ? "처리 중..." : "콘텐츠 생성"}
			</Button>
		</InputGroupWrapper>
	);
};

export default InputGroup;
