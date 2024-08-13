import React, { useState } from "react";
import styled from "styled-components";
import axios from "axios";
import InputGroup from "./components/InputGroup";
import ResultDisplay from "./components/ResultDisplay";
import LoadingSpinner from "./components/LoadingSpinner";

const AppContainer = styled.div`
	padding: 40px 20px;
	max-width: 800px;
	margin: 0 auto;
	font-family: "Helvetica Neue", Arial, sans-serif;
`;

const Header = styled.h1`
	text-align: center;
	color: #333;
	font-size: 2.5em;
	margin-bottom: 20px;
`;

const Description = styled.p`
	text-align: center;
	margin-bottom: 40px;
	color: #666;
	font-size: 1.2em;
`;

const Error = styled.div`
	color: red;
	text-align: center;
	margin-top: 20px;
	font-size: 1.1em;
`;

function App() {
	const [youtubeUrl, setYoutubeUrl] = useState("");
	const [result, setResult] = useState(null);
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState("");

	const handleInputChange = (event) => {
		setYoutubeUrl(event.target.value);
	};

	const fetchData = async () => {
		setLoading(true);
		setError("");
		setResult(null);

		try {
			const response = await axios.post("http://localhost:8100/crewai", {
				youtube_url: youtubeUrl,
			});
			setResult(response.data);
		} catch (err) {
			setError(err.response ? err.response.data.error : err.message);
		} finally {
			setLoading(false);
		}
	};

	const handleDownload = () => {
		const element = document.createElement("a");
		const file = new Blob([result.raw], { type: "text/plain" });
		element.href = URL.createObjectURL(file);
		element.download = "raw_data.txt";
		document.body.appendChild(element);
		element.click();
	};

	return (
		<AppContainer>
			<Header>CrewAI 콘텐츠 생성기</Header>
			<Description>YouTube 비디오에서 콘텐츠를 생성하기 위해 CrewAI의 강력한 백엔드를 사용하세요.</Description>

			{loading ? (
				<LoadingSpinner />
			) : (
				<>
					<InputGroup youtubeUrl={youtubeUrl} handleInputChange={handleInputChange} fetchData={fetchData} loading={loading} />

					{error && <Error>{error}</Error>}

					<ResultDisplay result={result} handleDownload={handleDownload} />
				</>
			)}
		</AppContainer>
	);
}

export default App;
