// 텍스트 이미지 VQA 솔루션 설명 PPT (A084_성훈 있는 성훈팀)
// 근거 문서: "텍스트 이미지 VQA — 데이터 처리 및 모델 개발 과정" + 프로젝트 실험 기록
const pptxgen = require('pptxgenjs');
const path = require('path');
const fs = require('fs');

const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE'; // 13.333 x 7.5 in
pres.author = 'A084_성훈 있는 성훈팀';
pres.company = 'SSAFY 16기';
pres.title = '텍스트 이미지 VQA 솔루션 설명';

const F = 'Malgun Gothic';
const C = {
  navy: '14213D', navy2: '22335A', navy3: '2B3F69', amber: 'FCA311', amberTint: 'FFF1D6',
  light: 'F1F3F7', line: 'D5DAE3', gray: '525C6E', muted: '8C95A6', white: 'FFFFFF', ice: 'C9D3E6',
};
const W = 13.333, M = 0.6, CW = W - 2 * M;
const TEAM = 'A084_성훈 있는 성훈팀';

// ---------- helpers ----------
function T(slide, text, o) {
  slide.addText(text, Object.assign({ fontFace: F, color: C.navy, margin: 0, isTextBox: true, valign: 'top' }, o));
}
function title(slide, t, lead) {
  T(slide, t, { x: M, y: 0.42, w: CW, h: 0.72, fontSize: 30, bold: true, valign: 'middle' });
  if (lead) T(slide, lead, { x: M, y: 1.14, w: CW, h: 0.42, fontSize: 15, color: C.gray });
}
function footer(slide, n, dark) {
  const col = dark ? '8793AD' : C.muted;
  T(slide, `${TEAM} · 텍스트 이미지 VQA 솔루션`, { x: M, y: 7.06, w: 8, h: 0.26, fontSize: 9, color: col });
  T(slide, String(n), { x: W - M - 1, y: 7.06, w: 1, h: 0.26, fontSize: 9, color: col, align: 'right' });
}
function tile(slide, x, y, s, label, o = {}) {
  const fill = o.fill || C.amber;
  slide.addText(label, {
    shape: pres.shapes.ROUNDED_RECTANGLE, x, y, w: s, h: s, rectRadius: s * 0.2,
    fill: { color: fill }, line: { color: o.lineColor || fill, width: 0.75 },
    fontFace: F, fontSize: o.fontSize || 16, bold: true, color: o.color || C.navy,
    align: 'center', valign: 'middle', margin: 0, isTextBox: true,
  });
}
function box(slide, x, y, w, h, o = {}) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: o.r === undefined ? 0.1 : o.r,
    fill: { color: o.fill || C.light }, line: { color: o.line || o.fill || C.light, width: o.lw || 0.75 },
  });
}
function arrow(slide, x1, y1, x2, y2, color) {
  slide.addShape(pres.shapes.LINE, {
    x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
    flipH: x2 < x1, flipV: y2 < y1,
    line: { color: color || C.muted, width: 1.5, endArrowType: 'triangle' },
  });
}
// 카드: 번호 타일 + 굵은 제목 + 본문
function infoCard(slide, x, y, w, h, head, body, o = {}) {
  box(slide, x, y, w, h, { fill: o.fill || C.light, line: o.line });
  let tx = x + 0.25, tw = w - 0.5;
  if (o.num !== undefined) {
    tile(slide, x + 0.25, y + 0.24, 0.46, String(o.num), { fontSize: 15, fill: o.tileFill, color: o.tileColor });
    tx = x + 0.88; tw = w - 1.13;
  }
  if (o.compact) {
    T(slide, head, { x: tx, y: y + 0.14, w: tw, h: 0.42, fontSize: o.headSize || 16, bold: true, color: o.headColor || C.navy, valign: 'middle' });
    T(slide, body, { x: x + 0.25, y: y + 0.6, w: w - 0.5, h: h - 0.7, fontSize: o.bodySize || 13, color: o.bodyColor || C.gray, lineSpacingMultiple: 1.1 });
    return;
  }
  T(slide, head, { x: tx, y: y + 0.22, w: tw, h: 0.5, fontSize: o.headSize || 16, bold: true, color: o.headColor || C.navy, valign: 'middle' });
  T(slide, body, { x: x + 0.25, y: y + 0.84, w: w - 0.5, h: h - 1.0, fontSize: o.bodySize || 13, color: o.bodyColor || C.gray, lineSpacingMultiple: 1.12 });
}
const tableOpts = (o) => Object.assign({
  fontFace: F, fontSize: 12, color: C.navy, valign: 'middle',
  border: { type: 'solid', pt: 0.75, color: C.line }, margin: [0.05, 0.1, 0.05, 0.1],
}, o);
const th = (text, o = {}) => ({ text, options: Object.assign({ bold: true, fill: { color: C.navy }, color: C.white, fontSize: 12 }, o) });
const td = (text, o = {}) => ({ text, options: o });

// =========================================================
// 1. 표지
// =========================================================
{
  const s = pres.addSlide();
  s.background = { color: C.navy };
  T(s, 'SSAFY 16기 2회차 AI 챌린지 · 텍스트 이미지 기반 VQA', { x: 0.75, y: 0.7, w: 7.6, h: 0.4, fontSize: 15, color: C.ice });
  T(s, '여러 VLM의 a~d 확률을\n검증으로 묶다', { x: 0.75, y: 1.55, w: 7.6, h: 2.1, fontSize: 42, bold: true, color: C.white, lineSpacingMultiple: 1.05 });
  T(s, '이미지 속 글자를 읽는 4지선다 VQA 솔루션 설명\n접근 전략 · 데이터 처리 · 모델 개발 · 실험 설계 · 실행 방법',
    { x: 0.75, y: 3.85, w: 7.6, h: 0.95, fontSize: 16, color: C.ice, lineSpacingMultiple: 1.2 });
  T(s, TEAM, { x: 0.75, y: 5.55, w: 7.6, h: 0.5, fontSize: 22, bold: true, color: C.amber });
  T(s, '2026-09-28', { x: 0.75, y: 6.08, w: 7.6, h: 0.35, fontSize: 14, color: C.ice });
  // 오른쪽: 답 타일 a~d (a = 정답 자리)
  const letters = ['a', 'b', 'c', 'd'];
  const s0 = 1.8, gap = 0.28, x0 = 8.75, y0 = 1.55;
  letters.forEach((L, i) => {
    const x = x0 + (i % 2) * (s0 + gap), y = y0 + Math.floor(i / 2) * (s0 + gap);
    const on = i === 0;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: s0, h: s0, rectRadius: 0.3, fill: { color: on ? C.amber : C.navy2 }, line: { color: on ? C.amber : C.navy3, width: 1 } });
    T(s, L, { x, y: y + 0.18, w: s0, h: 1.0, fontSize: 54, bold: true, align: 'center', valign: 'middle', color: on ? C.navy : C.white });
    T(s, `p(${L})`, { x, y: y + 1.18, w: s0, h: 0.4, fontSize: 15, align: 'center', color: on ? C.navy : C.ice });
  });
  T(s, '답 자리 토큰의 a~d 확률로 답을 고릅니다', { x: x0, y: y0 + 2 * s0 + gap + 0.18, w: 2 * s0 + gap, h: 0.35, fontSize: 12, color: C.ice, align: 'center' });
  s.addNotes('SSAFY 16기 2회차 AI 챌린지 A084 성훈 있는 성훈팀의 솔루션입니다. 핵심은 여러 비전-언어 모델이 낸 a~d 확률을, 공개 점수가 아니라 고정 검증 1,000문항만으로 묶은 것입니다.');
}

// =========================================================
// 2. 한눈에 보기
// =========================================================
{
  const s = pres.addSlide();
  title(s, '한눈에 보기', '여러 VLM을 같은 조건으로 학습·채점하고, 검증 1,000문항만으로 조합을 골랐습니다.');
  const stats = [
    ['검증 정확도', '98.0%', '최종 조합 · 고정 검증 1,000문항'],
    ['공개 리더보드', '0.97736', '최고 점수 · 3,281 / 3,357문항'],
    ['학습 규모', '20개', 'LoRA 실행 · 기반 모델 10종(7계열)'],
  ];
  const cw = (CW - 2 * 0.3) / 3;
  stats.forEach(([cap, big, sub], i) => {
    const x = M + i * (cw + 0.3), y = 1.85;
    box(s, x, y, cw, 2.05, { fill: i === 1 ? C.navy : C.light });
    const dark = i === 1;
    T(s, cap, { x: x + 0.3, y: y + 0.22, w: cw - 0.6, h: 0.35, fontSize: 13, bold: true, color: dark ? C.amber : C.gray });
    T(s, big, { x: x + 0.3, y: y + 0.58, w: cw - 0.6, h: 0.85, fontSize: 42, bold: true, color: dark ? C.white : C.navy, valign: 'middle' });
    T(s, sub, { x: x + 0.3, y: y + 1.5, w: cw - 0.6, h: 0.4, fontSize: 13, color: dark ? C.ice : C.gray });
  });
  const strat = [
    ['같은 조건으로 비교', '고정 검증 분할 · 같은 프롬프트 · 답 자리 a~d 확률 채점으로 모든 모델을 같은 문항에서 직접 비교했습니다.'],
    ['다양한 모델 확보', 'VLM 10종을 LoRA로 파인튜닝하고, 학습 없는 대형 모델(Qwen3.5-122B GPTQ 등)까지 모든 확률을 저장했습니다.'],
    ['검증으로만 조합 선택', '확률 평균 조합을 검증 1,000문항으로 고르고, 고르는 과정의 부풀림은 중첩 교차검증으로 따로 쟀습니다.'],
  ];
  strat.forEach(([h, b], i) => {
    const x = M + i * (cw + 0.3);
    infoCard(s, x, 4.25, cw, 2.5, h, b, { num: i + 1, bodySize: 14, headSize: 17, fill: C.white, line: C.line });
  });
  footer(s, 2);
  s.addNotes('단독 모델은 검증 95.7~97.7%였고, 확률 평균 조합이 98.0%로 가장 높았습니다. 이 조합의 공개 점수가 0.97736입니다. 학습 실행은 20개, 기반 모델은 10종(7계열)입니다.');
}

// =========================================================
// 3. 과제와 데이터
// =========================================================
{
  const s = pres.addSlide();
  title(s, '과제와 데이터', '이미지 속 글자를 읽고 네 보기 중 정답 하나를 고르는 과제 — 평가 지표는 정확도');
  // 대회 이미지(train_0001.jpg)는 공개 저장소에 넣지 않음 → 파일이 없으면 같은 자리에 빈 틀만 그림
  const IMG = path.join(__dirname, 'train_0001.jpg');
  if (fs.existsSync(IMG)) {
    s.addImage({ path: IMG, x: M, y: 1.8, w: 3.45, h: 4.6, altText: '대회 학습 이미지 예시: 선반 위 라이트박스에 SUNDAY BURGER LOVE' });
    T(s, 'train 예시 이미지', { x: M, y: 6.47, w: 3.45, h: 0.3, fontSize: 11, color: C.muted });
  } else {
    box(s, M, 1.8, 3.45, 4.6, { fill: C.white, line: C.line, lw: 1 });
    T(s, '예시 이미지 자리\n\n대회 데이터라\n공개본에는 넣지 않았습니다', { x: M + 0.3, y: 1.8, w: 2.85, h: 4.6, fontSize: 14, color: C.muted, align: 'center', valign: 'middle' });
    T(s, 'train 예시 이미지 (원본은 대회 데이터)', { x: M, y: 6.47, w: 3.45, h: 0.3, fontSize: 11, color: C.muted });
  }
  // 문제 카드
  const qx = 4.35, qw = 4.15;
  box(s, qx, 1.8, qw, 4.6, { fill: C.light });
  T(s, 'Q. 사진 속 선반에 적힌 문구는 무엇인가요?', { x: qx + 0.3, y: 2.0, w: qw - 0.6, h: 0.8, fontSize: 16, bold: true, lineSpacingMultiple: 1.1 });
  const opts = [['a', 'SUNDAY BURGER LOVE'], ['b', 'MONDAY SALAD TIME'], ['c', 'FRIDAY PIZZA PARTY'], ['d', 'WEDNESDAY SOUP DAY']];
  opts.forEach(([L, t], i) => {
    const y = 2.95 + i * 0.72;
    tile(s, qx + 0.3, y, 0.5, L, { fill: i === 0 ? C.amber : C.white, lineColor: i === 0 ? C.amber : C.line, fontSize: 16 });
    T(s, t, { x: qx + 0.98, y, w: qw - 1.25, h: 0.5, fontSize: 14, valign: 'middle', bold: i === 0 });
  });
  T(s, '정답: a', { x: qx + 0.3, y: 5.85, w: qw - 0.6, h: 0.35, fontSize: 13, color: C.gray });
  // 데이터 표
  const rows = [
    [th('데이터'), th('문항', { align: 'right' }), th('쓰임')],
    [td('train − 검증'), td('5,714', { align: 'right' }), td('LoRA 학습')],
    [td('고정 검증'), td('1,000', { align: 'right' }), td('모델 비교 · 조합 선택')],
    [td('test'), td('6,714', { align: 'right' }), td('추론만 (공개 점수는 3,357문항)')],
    [td('dev'), td('2,683', { align: 'right' }), td('정답 없음 → 조건부 의사 라벨')],
  ];
  s.addTable(rows, tableOpts({ x: 8.8, y: 1.8, w: 3.93, colW: [1.15, 0.78, 2.0], fontSize: 12, rowH: 0.52 }));
  T(s, [
    { text: '사진은 대부분 720×960', options: { bullet: true, breakLine: true } },
    { text: '외부 데이터는 쓰지 않음', options: { bullet: true, breakLine: true } },
    { text: 'train 6,714 = 학습 5,714 + 검증 1,000', options: { bullet: true } },
  ], { x: 8.8, y: 4.65, w: 3.93, h: 1.5, fontSize: 13, color: C.gray, paraSpaceAfter: 6 });
  footer(s, 3);
  s.addNotes('학습에는 대회 train만 썼고, 그중 1,000문항을 고정 검증셋으로 떼어 모든 모델과 조합을 같은 문항으로 비교했습니다. dev는 정답이 없어 조건을 만족한 문항만 의사 라벨로 썼습니다.');
}

// =========================================================
// 4. 접근 전략 — 전체 파이프라인
// =========================================================
{
  const s = pres.addSlide();
  title(s, '접근 전략 — 전체 파이프라인', '모든 모델이 같은 형식(문항별 a~d 확률)으로 예측을 남기고, 조합 단계에서 합칩니다.');
  const cx = [0.6, 3.05, 5.85, 8.3, 10.85], cwid = [2.0, 2.35, 2.0, 2.1, 1.88];
  const node = (i, y, h, head, body, o = {}) => {
    box(s, cx[i], y, cwid[i], h, { fill: o.fill || C.light, line: o.line || C.line });
    T(s, [{ text: head, options: { bold: true, fontSize: 14, color: o.hc || C.navy, breakLine: true } },
      { text: body, options: { fontSize: 11.5, color: o.bc || C.gray } }],
      { x: cx[i] + 0.15, y: y + 0.12, w: cwid[i] - 0.3, h: h - 0.24, valign: 'middle', align: 'center', lineSpacingMultiple: 1.1 });
  };
  const top = 1.85, full = 4.55;
  // col 1
  node(0, top, full, '대회 데이터', 'train 6,714\n→ 학습 5,714\n→ 고정 검증 1,000\n\ntest 6,714\n\nOCR 글자 (선택)\nPaddleOCR-VL');
  // col 2
  node(1, 1.85, 1.35, 'LoRA 파인튜닝', 'Gemma 4 · Qwen3.5 · Qwen3.8\nInternVL3.5 · HCX 등 (Colab)');
  node(1, 3.45, 1.35, '초대형 모델 LoRA', 'Qwen3.8-Flash-Next\n(RunPod B300)');
  node(1, 5.05, 1.35, '학습 없는 모델', 'Qwen3.5-122B GPTQ · Gemma 4\n(vLLM 채점)');
  // col 3
  node(2, top, full, '예측 저장', '검증 1,000 · test 6,714\n문항마다 a~d 확률\n\n평가 해상도 up2.0\n\nvqa_runs/<실행>/\n예측 CSV + train_log');
  // col 4
  node(3, 1.85, 2.1, '후보 A\n조합 탐색', '확률 평균\n중첩 CV로 방법 평가');
  node(3, 4.3, 2.1, '후보 B\n교체 규칙', '기준 모델 + 보조 표\n교차 적합 검증');
  // col 5
  node(4, top, full, '최종 제출 2개', '후보 A\nea0e9f (조합)\n\n후보 B\n505d49 (교체 규칙)', { fill: C.amber, line: C.amber, hc: C.navy, bc: C.navy });
  // arrows (세로로 긴 상자 사이는 수평 화살표)
  const r = (i) => cx[i] + cwid[i], l = (i) => cx[i];
  [2.525, 4.125, 5.725].forEach((y) => { arrow(s, r(0), y, l(1), y); arrow(s, r(1), y, l(2), y); });
  [2.9, 5.35].forEach((y) => { arrow(s, r(2), y, l(3), y); arrow(s, r(3), y, l(4), y); });
  footer(s, 4);
  s.addNotes('데이터에서 세 갈래로 모델을 만들고, 모두 검증 1,000문항과 test 6,714문항의 a~d 확률을 같은 형식으로 저장합니다. 이 확률을 후보 A(조합 탐색)와 후보 B(교체 규칙)로 합쳐 제출 파일을 만듭니다.');
}

// =========================================================
// 5. 실험 설계
// =========================================================
{
  const s = pres.addSlide();
  title(s, '실험 설계 — 무엇이 진짜 개선인지 가리기', '검증이 1,000문항(오답 20여 개)이라 ±3문항은 우연 범위 — 판정 절차부터 정했습니다.');
  const items = [
    ['고정 검증 1,000문항', 'train에서 정답 글자별 250문항씩 뽑음(시드 42). 모든 노트북이 분할 md5를 확인하고, 다르면 멈춥니다.', '글자별 250 × 4 = 1,000문항 · 분할 md5 df05e4bc…'],
    ['같은 문항 비교', '정확도 한 숫자 대신 고친 문항 · 새로 틀린 문항 · 순이득으로 판정하고 부호 검정을 봅니다.', '예: Qwen3.5 학습 전 → 후  고침 13 · 새로 틀림 2 → 순이득 +11'],
    ['규칙은 결과 전에', '채택 기준을 실행 전에 적어 둡니다. 예: 앙상블 순이득 ≥ +3이고 새로 틀린 문항 ≤ 고친 문항의 절반.', '예: 학습 변형 19가지 → 모두 기준 미달로 불채택'],
    ['고르는 과정도 평가', '조합·문턱값 선택은 중첩 교차검증·교차 적합으로 재고, 공개 리더보드는 확인용으로만 씁니다.', '예: 검증 1등 조합 98.0% → 같은 방법의 중첩 CV 97.53%'],
  ];
  const w2 = (CW - 0.3) / 2, h2 = 2.35;
  items.forEach(([h, b, ex], i) => {
    const x = M + (i % 2) * (w2 + 0.3), y = 1.85 + Math.floor(i / 2) * (h2 + 0.3);
    box(s, x, y, w2, h2, { fill: C.light });
    tile(s, x + 0.25, y + 0.24, 0.46, String(i + 1), { fontSize: 15 });
    T(s, h, { x: x + 0.88, y: y + 0.22, w: w2 - 1.13, h: 0.5, fontSize: 18, bold: true, valign: 'middle' });
    T(s, b, { x: x + 0.25, y: y + 0.82, w: w2 - 0.5, h: 0.72, fontSize: 14, color: C.gray, lineSpacingMultiple: 1.1 });
    box(s, x + 0.25, y + h2 - 0.72, w2 - 0.5, 0.5, { fill: C.white, line: C.line, r: 0.08 });
    T(s, ex, { x: x + 0.45, y: y + h2 - 0.72, w: w2 - 0.9, h: 0.5, fontSize: 12.5, bold: true, color: C.navy, valign: 'middle' });
  });
  footer(s, 5);
  s.addNotes('검증 1,000문항에서는 1문항이 0.1%p입니다. 오답이 20여 개뿐이라 몇 문항 차이는 우연일 수 있어, 같은 문항끼리 비교하고 채택 기준을 결과보다 먼저 정했습니다.');
}

// =========================================================
// 6. 데이터 처리
// =========================================================
{
  const s = pres.addSlide();
  title(s, '데이터 처리', '대회 데이터만 사용 — 해상도, 보기 섞기, 보조·의사 라벨, OCR, 라벨 점검');
  const items = [
    ['이미지 · 해상도', 'EXIF 방향 보정 · RGB. 평가·test는 up2.0(약 2배): Qwen·HCX 270만~340만 픽셀, Gemma 4 소프트 토큰 1,120개, InternVL 타일 12개.'],
    ['증강', 'epoch마다 보기 순서를 섞고 정답 글자도 함께 바꿈(자리 편향 제거). 글자 과제라 회전·반전·자르기는 쓰지 않음.'],
    ['보조 학습 데이터', '보기 없이도 답이 정해지는 글자 읽기 문항(가격·시간·전화·상호·층·문구)만 골라 단답형 목표로 사용.'],
    ['dev 의사 라벨', '교육생 응답은 4~5명 일치가 0문항(최다 3표). 모델 확신 ≥ 0.8이고 3표 이상 다수결과 같은 문항만 사용.'],
    ['OCR (선택)', 'PaddleOCR-VL로 글자 줄·위치 추출. 반복 루프는 정리하고, 1,000자를 넘으면 질문·보기와 겹치는 줄부터 남김.'],
    ['라벨 점검', '검증 오답을 이미지로 직접 확인: 라벨 오류 약 10 · 답이 둘 다 되는 문항 약 8. 학습 라벨은 원본 그대로 사용.'],
  ];
  const w3 = (CW - 2 * 0.3) / 3, h3 = 2.4;
  items.forEach(([h, b], i) => {
    const x = M + (i % 3) * (w3 + 0.3), y = 1.85 + Math.floor(i / 3) * (h3 + 0.3);
    infoCard(s, x, y, w3, h3, h, b, { num: i + 1, bodySize: 14 });
  });
  footer(s, 6);
  s.addNotes('해상도는 이름 붙인 프리셋으로 관리했고, 대회 사진이 대부분 720×960이라 평가는 약 2배로 키운 up2.0으로 통일했습니다. 라벨 오류가 있어도 test에 같은 성격의 문항이 있을 수 있어 학습 라벨은 고치지 않았습니다.');
}

// =========================================================
// 7. 기반 모델 고르기 (zero-shot 300문항)
// =========================================================
{
  const s = pres.addSlide();
  title(s, '기반 모델 고르기 — 학습 전에 먼저 재기', '검증 앞 300문항 zero-shot으로 한국어 글자 읽기를 비교하고, 약한 모델은 학습하지 않았습니다.');
  const data = [ // 아래 → 위 순서
    ['EXAONE-4.5-33B', 85.67, 'n'], ['GLM-5.3-Flash', 92.0, 'n'], ['A.X-4.0-VL-Light', 93.0, 'n'],
    ['InternVL3.5-38B', 94.33, 't'], ['HyperCLOVA X 32B', 95.67, 't'], ['Qwen3.6-35B-A3B', 96.33, 't'],
    ['Qwen3.5-27B', 96.33, 't'], ['Qwen3.8-Flash-Next', 96.67, 't'], ['Gemma-4-31B', 97.67, 'top'],
  ];
  const colors = data.map(([, , k]) => (k === 'n' ? 'B6BDCA' : k === 'top' ? C.amber : C.navy));
  s.addChart(pres.charts.BAR, [{ name: 'zero-shot 정확도', labels: data.map((d) => d[0]), values: data.map((d) => d[1]) }], {
    x: M, y: 1.75, w: 7.9, h: 5.15, barDir: 'bar', chartColors: colors, barGapWidthPct: 45,
    valAxisMinVal: 80, valAxisMaxVal: 100, valAxisMajorUnit: 5, valAxisLabelFormatCode: '0',
    showValue: true, dataLabelPosition: 'outEnd', dataLabelFormatCode: '0.00', dataLabelFontSize: 12, dataLabelColor: C.navy, dataLabelFontFace: F,
    catAxisLabelFontFace: F, catAxisLabelFontSize: 12, catAxisLabelColor: C.navy, valAxisLabelFontFace: F, valAxisLabelFontSize: 11, valAxisLabelColor: C.muted,
    valGridLine: { color: 'E3E7EE', size: 0.75 }, catGridLine: { style: 'none' }, showLegend: false,
    showTitle: true, title: 'zero-shot 정확도 (%) · 검증 앞 300문항 · up2.0', titleFontFace: F, titleFontSize: 13, titleColor: C.gray,
  });
  // 오른쪽 설명
  const rx = 8.85, rw = W - M - rx;
  const legend = [[C.navy, '학습한 모델'], ['B6BDCA', '학습하지 않음 (글자 읽기 약함)'], [C.amber, 'zero-shot 1위']];
  legend.forEach(([col, lab], i) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: rx, y: 1.95 + i * 0.42, w: 0.26, h: 0.26, rectRadius: 0.05, fill: { color: col }, line: { color: col } });
    T(s, lab, { x: rx + 0.4, y: 1.92 + i * 0.42, w: rw - 0.4, h: 0.32, fontSize: 13, valign: 'middle' });
  });
  box(s, rx, 3.35, rw, 3.35, { fill: C.light });
  T(s, [
    { text: '학습 후 대략 +1.1~1.4%p', options: { bold: true, color: C.navy, breakLine: true } },
    { text: '예: Qwen3.5-27B 96.1 → 97.2 (같은 1,000문항)', options: { breakLine: true } },
    { text: ' ', options: { fontSize: 6, breakLine: true } },
    { text: '크기보다 계열', options: { bold: true, color: C.navy, breakLine: true } },
    { text: '320B GLM-5.3-Flash 92.0 · 122B zero-shot 96.0(1,000문항) < Gemma 4 31B', options: { breakLine: true } },
    { text: ' ', options: { fontSize: 6, breakLine: true } },
    { text: '형식부터 확인', options: { bold: true, color: C.navy, breakLine: true } },
    { text: 'Muse-Glimmer-30B는 답 자리 형식이 달라 측정 무효 (letter_mass 0.003)' },
  ], { x: rx + 0.25, y: 3.55, w: rw - 0.5, h: 3.0, fontSize: 13, color: C.gray, lineSpacingMultiple: 1.12 });
  footer(s, 7);
  s.addNotes('후보 모델을 학습 없이 같은 300문항으로 먼저 재었습니다. Gemma 4 31B가 97.67%로 가장 높았고, EXAONE·GLM·A.X처럼 한국어 글자 읽기가 약한 모델은 학습하지 않았습니다. 모델 크기가 크다고 더 잘 읽지는 않았습니다.');
}

// =========================================================
// 8. 학습 레시피
// =========================================================
{
  const s = pres.addSlide();
  title(s, '모델 학습 — 하나의 레시피, 여러 모델', '통합 노트북 01_train_vlm에 실행 이름(RECIPE)만 넣으면 train_log와 같은 설정으로 재현됩니다.');
  const rows = [
    [th('항목'), th('설정')],
    [td('학습 목표', { bold: true }), td('추론과 똑같은 프롬프트의 마지막 위치(답 자리) 로짓에만 교차 엔트로피 · 목표는 정답 소문자 한 글자')],
    [td('LoRA', { bold: true }), td('r 16 · α 32 · dropout 0.05 · 언어 모델의 어텐션·MLP 선형층. 선택으로 비전 블록·투영층도(LR × 0.2)')],
    [td('최적화', { bold: true }), td('AdamW · LR 1e-4(이어학습 2e-5) · cosine + warmup 5% · 유효 배치 16 · 1 epoch (8B·9B는 2 epoch 중 검증으로 선택)')],
    [td('정밀도', { bold: true }), td('20B 이상은 4bit NF4(QLoRA) 또는 bf16. 비전 타워·투영층·lm_head는 양자화하지 않음')],
    [td('학습 해상도', { bold: true }), td('native (Gemma 4 31B만 up2.0이 더 좋아 up2.0)')],
    [td('이어하기', { bold: true }), td('50스텝마다 어댑터·옵티마이저·난수 상태 저장 → 끊긴 곳부터 같은 결과로 이어 학습')],
  ];
  s.addTable(rows, tableOpts({ x: M, y: 1.8, w: 7.75, colW: [1.45, 6.3], fontSize: 12.5, rowH: [0.45, 0.78, 0.78, 0.78, 0.7, 0.55, 0.7] }));
  // 오른쪽: 답 자리 학습 도식
  const rx = 8.7, rw = W - M - rx;
  box(s, rx, 1.8, rw, 4.75, { fill: C.light });
  T(s, '답 자리 한 곳만 학습', { x: rx + 0.25, y: 1.95, w: rw - 0.5, h: 0.4, fontSize: 16, bold: true });
  const toks = ['이미지', '질문', '(a)~(d) 보기', '“한 글자로 답”'];
  toks.forEach((t, i) => {
    box(s, rx + 0.3, 2.5 + i * 0.52, rw - 0.6, 0.42, { fill: C.white, line: C.line, r: 0.06 });
    T(s, t, { x: rx + 0.3, y: 2.5 + i * 0.52, w: rw - 0.6, h: 0.42, fontSize: 12.5, align: 'center', valign: 'middle', color: C.gray });
  });
  arrow(s, rx + rw / 2, 4.58, rx + rw / 2, 4.86, C.navy);
  s.addText('답 자리 → 정답 글자 1개', { shape: pres.shapes.ROUNDED_RECTANGLE, x: rx + 0.3, y: 4.9, w: rw - 0.6, h: 0.5, rectRadius: 0.08,
    fill: { color: C.amber }, line: { color: C.amber }, fontFace: F, fontSize: 13.5, bold: true, color: C.navy, align: 'center', valign: 'middle', margin: 0, isTextBox: true });
  T(s, '학습 뒤 letter_mass(답 자리의 a~d 확률 합)가 모든 모델에서 1.000', { x: rx + 0.3, y: 5.55, w: rw - 0.6, h: 0.85, fontSize: 12, color: C.gray, lineSpacingMultiple: 1.1 });
  footer(s, 8);
  s.addNotes('학습 목표는 추론과 같은 프롬프트의 답 자리 한 곳에서 정답 글자를 맞히는 것입니다. 그래서 학습 뒤에는 답 자리에 a~d 외의 토큰이 거의 나오지 않고, 확률을 그대로 조합에 쓸 수 있습니다.');
}

// =========================================================
// 9. 모델별 처리
// =========================================================
{
  const s = pres.addSlide();
  title(s, '모델별 처리 — 답 자리와 하드웨어 맞추기', '템플릿·해상도 단위·커널이 모델마다 달라, 같은 채점을 하려면 개별 보정이 필요했습니다.');
  const rows = [
    ['HCX', 'HyperCLOVA X SEED Think 32B', '그대로 두면 답 자리에서 생각 블록을 시작 → thinking=False를 넘기고 열린 <think>를 </think>로 닫아 다음 토큰이 곧 답 글자가 되게 함 (transformers 개발 버전 커밋)'],
    ['해상도', 'Gemma 4 · InternVL3.5 · Qwen3.5', '해상도 단위가 다름: 이미지당 소프트 토큰 수 / 448px 타일 수 / 픽셀 범위. Qwen3.5 이후는 선형 어텐션 커널(flash-linear-attention)을 설치해야 OOM이 나지 않음'],
    ['B300', 'Qwen3.8-Flash-Next\n(bf16 330GiB)', '패딩 자리의 전부 가려진 어텐션 행 때문에 역전파가 NaN → 학습 중에만 풀어 넘김 · 희소 어텐션 토큰 선택 벡터화 · 95GiB n-gram 표는 CPU RAM에'],
    ['vLLM', '학습 없는 대형 모델 (vLLM 0.30)', 'Qwen3.5-122B는 사전 양자화 GPTQ-Int4로 G4 96GB 한 장에서 · B300에서는 즉석 컴파일이 안 돼 Triton 어텐션·MoE 커널 사용'],
  ];
  rows.forEach(([tag, head, body], i) => {
    const y = 1.8 + i * 1.18;
    box(s, M, y, CW, 1.05, { fill: C.light });
    s.addText(tag, { shape: pres.shapes.ROUNDED_RECTANGLE, x: M + 0.22, y: y + 0.225, w: 1.15, h: 0.6, rectRadius: 0.1, fill: { color: C.navy }, line: { color: C.navy },
      fontFace: F, fontSize: 13, bold: true, color: C.white, align: 'center', valign: 'middle', margin: 0, isTextBox: true });
    T(s, head, { x: M + 1.6, y: y + 0.08, w: 3.2, h: 0.89, fontSize: 14, bold: true, valign: 'middle' });
    T(s, body, { x: M + 4.9, y: y + 0.08, w: CW - 5.1, h: 0.89, fontSize: 12.5, color: C.gray, valign: 'middle', lineSpacingMultiple: 1.08 });
  });
  T(s, '보정마다 작은 모델로 원래 구현과 결과(로짓·손실·기울기)가 같은지 대조한 뒤 썼습니다.', { x: M, y: 6.58, w: CW, h: 0.3, fontSize: 12, color: C.muted });
  footer(s, 9);
  s.addNotes('HyperCLOVA X는 생각 모드 모델이라 답 자리에서 바로 글자를 내도록 빈 생각 블록을 닫았습니다. Flash-Next는 B300 한 장에서 학습하면서 역전파 NaN, 느린 희소 어텐션, 큰 임베딩 표 문제를 각각 고쳤습니다.');
}

// =========================================================
// 10. 추론
// =========================================================
{
  const s = pres.addSlide();
  title(s, '추론 — 생성하지 않고 답 자리 확률로', '모든 모델이 문항마다 a~d 네 확률을 남기고, 이 확률이 조합의 재료가 됩니다.');
  // 왼쪽 도식
  const lx = M, lw = 6.1;
  box(s, lx, 1.8, lw, 4.95, { fill: C.light });
  const step = (y, t) => {
    box(s, lx + 0.35, y, lw - 0.7, 0.55, { fill: C.white, line: C.line, r: 0.08 });
    T(s, t, { x: lx + 0.35, y, w: lw - 0.7, h: 0.55, fontSize: 13, align: 'center', valign: 'middle' });
  };
  step(2.0, '이미지 + 질문 + (a)~(d) + “한 글자로 답하라”');
  arrow(s, lx + lw / 2, 2.57, lx + lw / 2, 2.8, C.navy);
  step(2.82, 'VLM (LoRA 학습 또는 zero-shot)');
  arrow(s, lx + lw / 2, 3.39, lx + lw / 2, 3.62, C.navy);
  T(s, '답 자리 다음 토큰의 확률', { x: lx + 0.35, y: 3.65, w: lw - 0.7, h: 0.35, fontSize: 13, bold: true, align: 'center' });
  const bars = [['a', 4.2], ['b', 0.35], ['c', 0.2], ['d', 0.28]]; // 개념 그림 (실제 값 아님)
  bars.forEach(([L, len], i) => {
    const y = 4.12 + i * 0.47;
    tile(s, lx + 0.5, y, 0.36, L, { fill: i === 0 ? C.amber : C.white, lineColor: i === 0 ? C.amber : C.line, fontSize: 13 });
    s.addShape(pres.shapes.RECTANGLE, { x: lx + 1.0, y: y + 0.09, w: len, h: 0.18, fill: { color: i === 0 ? C.amber : 'B6BDCA' }, line: { color: i === 0 ? C.amber : 'B6BDCA' } });
  });
  T(s, '대소문자·앞 공백 변형(a, A, “ a”, “ A”) 확률을 합쳐 네 보기로 정규화 · letter_mass(네 글자 확률 합)로 답 자리 점검', { x: lx + 0.35, y: 6.05, w: lw - 0.7, h: 0.6, fontSize: 11.5, color: C.gray, lineSpacingMultiple: 1.08 });
  // 오른쪽 카드 3개
  const rx = 7.0, rw = W - M - rx;
  const cards = [
    ['해상도', '검증·test 모두 up2.0. 보기 순열 4가지 평균(TTA)과 애매한 문항 고해상도(3·4배) 재채점은 이득이 없어 쓰지 않음.'],
    ['학습 없는 대형 모델 (vLLM)', '답 자리 상위 64개 로그확률에서 a~d 수집. 이미지 크기를 transformers 전처리와 맞춤. 검증 122B GPTQ 96.0% · Gemma 4 97.0%.'],
    ['판정 모델 (교차 검증)', '기준·보조 모델의 답이 갈린 문항만. 두 답을 모델 A/B로 제시하고 단계별 결론 → 표본 5개 다수결이 한 표, 부족·동점은 기권.'],
  ];
  cards.forEach(([h, b], i) => infoCard(s, rx, 1.8 + i * 1.69, rw, 1.57, h, b, { bodySize: 12.5, headSize: 15, compact: true }));
  footer(s, 10);
  s.addNotes('모든 모델은 답을 문장으로 생성하지 않고, 답 자리 토큰 하나의 a~d 확률로 답합니다. 왼쪽 막대는 개념 그림입니다. 대형 모델은 vLLM으로 같은 방식으로 채점했고, 답이 갈린 문항에는 판정 모델을 따로 돌렸습니다.');
}

// =========================================================
// 11. 단독 모델 성적
// =========================================================
{
  const s = pres.addSlide();
  title(s, '실험 결과 — 단독 95.7~97.7%, 조합 98.0%', '고정 검증 1,000문항 · up2.0. 실행 5개(4계열)의 같은 가중치 확률 평균이 가장 높았습니다.');
  const data = [ // 아래 → 위
    ['InternVL3.5-38B', 95.7, 't'], ['Qwen3.5-122B (zero-shot)', 96.0, 'z'], ['Qwen3.5-9B', 96.4, 't'], ['Qwen3-VL-8B', 96.5, 't'],
    ['Qwen3.8-27B', 96.8, 't'], ['Qwen3.8-Flash-Next', 96.9, 't'], ['Gemma-4-26B-A4B', 96.9, 't'], ['Gemma-4-31B (zero-shot)', 97.0, 'z'],
    ['Qwen3.6-35B-A3B', 97.0, 't'], ['HyperCLOVA X 32B', 97.0, 't'], ['Qwen3.5-27B', 97.2, 't'], ['Qwen3.5-27B 이어학습+보조', 97.6, 't'],
    ['Gemma-4-31B', 97.6, 't'], ['Gemma-4-31B OCR', 97.7, 't'], ['조합 ea0e9f (5개 평균)', 98.0, 'e'],
  ];
  const colors = data.map(([, , k]) => (k === 'z' ? 'B6BDCA' : k === 'e' ? C.amber : C.navy));
  s.addChart(pres.charts.BAR, [{ name: '검증 정확도', labels: data.map((d) => d[0]), values: data.map((d) => d[1]) }], {
    x: M, y: 1.7, w: 8.35, h: 5.3, barDir: 'bar', chartColors: colors, barGapWidthPct: 35,
    valAxisMinVal: 94, valAxisMaxVal: 98.5, valAxisMajorUnit: 1, valAxisLabelFormatCode: '0',
    showValue: true, dataLabelPosition: 'outEnd', dataLabelFormatCode: '0.0', dataLabelFontSize: 11, dataLabelColor: C.navy, dataLabelFontFace: F,
    catAxisLabelFontFace: F, catAxisLabelFontSize: 11, catAxisLabelColor: C.navy, valAxisLabelFontFace: F, valAxisLabelFontSize: 11, valAxisLabelColor: C.muted,
    valGridLine: { color: 'E3E7EE', size: 0.75 }, catGridLine: { style: 'none' }, showLegend: false,
    showTitle: true, title: '검증 정확도 (%) · 축은 94%부터', titleFontFace: F, titleFontSize: 12, titleColor: C.gray,
  });
  const rx = 9.3, rw = W - M - rx;
  const legend = [[C.navy, 'LoRA 학습 모델'], ['B6BDCA', 'zero-shot (학습 없음)'], [C.amber, '확률 평균 조합']];
  legend.forEach(([col, lab], i) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: rx, y: 1.95 + i * 0.42, w: 0.26, h: 0.26, rectRadius: 0.05, fill: { color: col }, line: { color: col } });
    T(s, lab, { x: rx + 0.4, y: 1.92 + i * 0.42, w: rw - 0.4, h: 0.32, fontSize: 13, valign: 'middle' });
  });
  box(s, rx, 3.35, rw, 3.4, { fill: C.light });
  T(s, [
    { text: '조합 멤버', options: { bold: true, color: C.navy, breakLine: true } },
    { text: 'Gemma 4 bf16 · Gemma 4 OCR · InternVL3.5 · Qwen3.5 bf16 · Qwen3.8', options: { breakLine: true } },
    { text: ' ', options: { fontSize: 6, breakLine: true } },
    { text: '단독 최하위도 조합에', options: { bold: true, color: C.navy, breakLine: true } },
    { text: 'InternVL(95.7)은 틀리는 문항이 달라 조합에 보탬', options: { breakLine: true } },
    { text: ' ', options: { fontSize: 6, breakLine: true } },
    { text: '정밀도는 병목 아님', options: { bold: true, color: C.navy, breakLine: true } },
    { text: '같은 시드 NF4·bf16 답 99% 동일 (Qwen3.5 97.2 / 97.1)' },
  ], { x: rx + 0.25, y: 3.55, w: rw - 0.5, h: 3.05, fontSize: 12.5, color: C.gray, lineSpacingMultiple: 1.1 });
  footer(s, 11);
  s.addNotes('단독 모델은 95.7~97.7% 사이에 몰려 있습니다. 조합의 98.0%는 조합을 고른 그 1,000문항에서 잰 값이라 조금 부풀려져 있고, 같은 방법의 중첩 교차검증은 97.53%였습니다. 축은 94%부터 시작합니다.');
}

// =========================================================
// 12. 후보 A — 조합 검색
// =========================================================
{
  const s = pres.addSlide();
  title(s, '후보 A — 검증으로 조합 고르기', '결합 3가지 × 검색 3가지로 조합을 찾고, 고르는 과정의 과적합은 중첩 교차검증으로 따로 쟀습니다.');
  const steps = [
    ['멤버', '고정 분할로 학습하고 검증·test 예측이 모두 있는 실행. 검증을 학습에 쓴 _full · 검증 99% 초과 실행은 제외'],
    ['결합', '확률 평균 / 멤버별 온도 보정 후 평균 / 로그확률 평균 (온도는 검증 NLL 최소)'],
    ['검색', '같은 가중치 부분집합 전수(설정마다 약 70만 조합) · 그리디 · 좌표 상승'],
    ['평가', '중첩 CV 5겹 × 4번: 800문항으로 검색하고 남은 200문항으로 채점'],
  ];
  steps.forEach(([h, b], i) => {
    const y = 1.85 + i * 1.18;
    tile(s, M, y, 0.5, String(i + 1), { fontSize: 16 });
    T(s, h, { x: M + 0.68, y, w: 1.2, h: 0.5, fontSize: 16, bold: true, valign: 'middle' });
    T(s, b, { x: M + 1.8, y: y - 0.02, w: 4.45, h: 1.05, fontSize: 13, color: C.gray, lineSpacingMultiple: 1.1 });
  });
  // 오른쪽: 부풀림
  const rx = 7.05, rw = W - M - rx;
  box(s, rx, 1.85, rw, 2.55, { fill: C.light });
  T(s, '98.0%', { x: rx + 0.3, y: 2.05, w: 2.3, h: 0.9, fontSize: 40, bold: true, valign: 'middle' });
  T(s, '검증 1,000문항에서\n고른 최고 조합', { x: rx + 0.3, y: 2.95, w: 2.3, h: 0.8, fontSize: 12.5, color: C.gray });
  arrow(s, rx + 2.65, 2.5, rx + 3.1, 2.5, C.navy);
  T(s, '97.53%', { x: rx + 3.25, y: 2.05, w: 2.3, h: 0.9, fontSize: 40, bold: true, color: C.navy, valign: 'middle' });
  T(s, '같은 방법의\n중첩 CV 정확도', { x: rx + 3.25, y: 2.95, w: 2.3, h: 0.8, fontSize: 12.5, color: C.gray });
  T(s, '→ 1,000문항에서 고른 최고값은 약 0.5%p 부풀려짐', { x: rx + 0.3, y: 3.8, w: rw - 0.6, h: 0.4, fontSize: 13, bold: true, color: C.navy });
  // 제출 조합
  T(s, '최종 제출 ea0e9f — 같은 가중치 확률 평균', { x: rx, y: 4.65, w: rw, h: 0.4, fontSize: 15, bold: true });
  const mem = ['Gemma 4 bf16', 'Gemma 4 OCR', 'InternVL3.5', 'Qwen3.5 bf16', 'Qwen3.8'];
  const mw = (rw - 4 * 0.12) / 5;
  mem.forEach((m, i) => {
    s.addText(m, { shape: pres.shapes.ROUNDED_RECTANGLE, x: rx + i * (mw + 0.12), y: 5.15, w: mw, h: 0.62, rectRadius: 0.08, fill: { color: C.navy }, line: { color: C.navy },
      fontFace: F, fontSize: 11.5, bold: true, color: C.white, align: 'center', valign: 'middle', margin: 0, isTextBox: true });
  });
  T(s, [{ text: '검증 98.0% · 공개 0.97736', options: { bold: true, color: C.navy } }, { text: ' (3,281 / 3,357) · 실행 5개 · 4계열' }],
    { x: rx, y: 5.95, w: rw, h: 0.4, fontSize: 13, color: C.gray });
  footer(s, 12);
  s.addNotes('조합 수십만 개를 1,000문항으로 고르면 1등은 운이 섞여 부풀려집니다. 그래서 800문항으로 고르고 200문항으로 채점하는 중첩 교차검증으로 방법 자체의 성능을 따로 쟀고, 차이는 약 0.5%p였습니다.');
}

// =========================================================
// 13. 후보 B — 교체 규칙
// =========================================================
{
  const s = pres.addSlide();
  title(s, '후보 B — 기준 모델 교체 규칙', '기준 모델(Flash-Next 학습 모델)의 답은 네 조건을 모두 만족할 때만 보조 모델 쪽으로 바꿉니다.');
  const conds = [
    ['불일치', '보조 모델이 기준과 다른 보기를 고름'],
    ['확신 낮음', '기준의 최고 확률 < CONF'],
    ['마진 작음', '기준의 1·2위 확률 차 < MARGIN'],
    ['합의', '보조 AGREE개 이상이 같은 다른 보기를 고름'],
  ];
  const cw4 = (CW - 3 * 0.25) / 4;
  conds.forEach(([h, b], i) => infoCard(s, M + i * (cw4 + 0.25), 1.8, cw4, 1.75, h, b, { num: i + 1, bodySize: 13, headSize: 16 }));
  // 채택 기준
  const lw = 4.6;
  box(s, M, 3.85, lw, 2.95, { fill: C.light });
  T(s, '채택 기준 (결과 전에 고정)', { x: M + 0.25, y: 3.98, w: lw - 0.5, h: 0.4, fontSize: 15, bold: true });
  T(s, [
    { text: '순이득 ≥ +10', options: { bullet: true, breakLine: true } },
    { text: '살린 문항 ≥ 망친 문항 × 2', options: { bullet: true, breakLine: true } },
    { text: '부트스트랩 1,000번 중 95% 이상에서 이득', options: { bullet: true, breakLine: true } },
    { text: '5구역에서 고른 문턱값이 비슷함', options: { bullet: true } },
  ], { x: M + 0.25, y: 4.45, w: lw - 0.5, h: 1.6, fontSize: 13, color: C.gray, paraSpaceAfter: 4 });
  T(s, '문턱값은 5겹 교차 적합으로 선택 · 기준 확률은 온도 보정 후 비교', { x: M + 0.25, y: 6.1, w: lw - 0.5, h: 0.6, fontSize: 11.5, color: C.muted, lineSpacingMultiple: 1.08 });
  // 결과 표
  const tx = M + lw + 0.3, tw = W - M - tx;
  const rows = [
    [th('실행'), th('보조 모델'), th('살림', { align: 'right' }), th('망침', { align: 'right' }), th('순이득', { align: 'right' }), th('부트스트랩', { align: 'right' }), th('판정')],
    [td('a12cf3'), td('학습 4 + zero-shot 2'), td('7', { align: 'right' }), td('3', { align: 'right' }), td('+4', { align: 'right' }), td('88%', { align: 'right' }), td('불채택')],
    [td('19f20f'), td('학습 4 + zero-shot 1'), td('6', { align: 'right' }), td('4', { align: 'right' }), td('+2', { align: 'right' }), td('70%', { align: 'right' }), td('불채택')],
  ];
  s.addTable(rows, tableOpts({ x: tx, y: 3.85, w: tw, colW: [1.0, 1.95, 0.62, 0.62, 0.78, 1.05, tw - 6.02], fontSize: 12, rowH: 0.5 }));
  box(s, tx, 5.55, tw, 1.25, { fill: C.amberTint, line: C.amberTint });
  T(s, [
    { text: '최종 제출 두 번째 칸: 505d49 (공개 0.97378)', options: { bold: true, color: C.navy, breakLine: true } },
    { text: '기준 단독(0.97110)보다 9문항 높음. 후보 A와 방식이 다른 파일로 둘째 칸을 채움 — 최종 점수는 두 제출 중 높은 쪽' },
  ], { x: tx + 0.25, y: 5.67, w: tw - 0.5, h: 1.05, fontSize: 12.5, color: C.gray, lineSpacingMultiple: 1.1 });
  footer(s, 13);
  s.addNotes('기준 모델이 불확실하고 보조 모델들의 표가 한 보기로 모일 때만 답을 바꿉니다. 교차 적합 검증에서는 순이득 +4, +2로 합격선(+10)에 못 미쳐 불채택이었습니다. 다만 최종 점수는 두 제출 중 높은 쪽이라, 후보 A와 방식이 다른 교체 규칙 파일 505d49를 두 번째 제출 칸에 넣었습니다.');
}

// =========================================================
// 14. 제출 결과
// =========================================================
{
  const s = pres.addSlide();
  title(s, '제출 결과와 최종 제출', '최종 제출은 후보 A 조합 ea0e9f와 후보 B 교체 규칙 505d49 — 공개 점수는 확인용으로만 썼습니다.');
  const hl = { fill: { color: C.amberTint }, bold: true };
  const rows = [
    [th('제출'), th('방법'), th('검증 %', { align: 'right' }), th('공개 점수', { align: 'right' }), th('공개 문항', { align: 'right' }), th('최종', { align: 'center' })],
    [td('ea0e9f (후보 A)', hl), td('Gemma 4 bf16 · Gemma 4 OCR · InternVL3.5 · Qwen3.5 bf16 · Qwen3.8, 확률 평균', hl), td('98.0', Object.assign({ align: 'right' }, hl)), td('0.97736', Object.assign({ align: 'right' }, hl)), td('3,281', Object.assign({ align: 'right' }, hl)), td('제출', Object.assign({ align: 'center' }, hl))],
    [td('505d49 (후보 B)', hl), td('Flash-Next 기준 + 보조 모델 표 (교체 규칙)', hl), td('—', Object.assign({ align: 'right' }, hl)), td('0.97378', Object.assign({ align: 'right' }, hl)), td('3,269', Object.assign({ align: 'right' }, hl)), td('제출', Object.assign({ align: 'center' }, hl))],
    [td('b47d7c'), td('ea0e9f 조합 + Flash-Next'), td('97.6', { align: 'right' }), td('0.97676', { align: 'right' }), td('3,279', { align: 'right' }), td('')],
    [td('majority_vote'), td('초기 조합 3개의 다수결'), td('—', { align: 'right' }), td('0.97676', { align: 'right' }), td('3,279', { align: 'right' }), td('')],
    [td('clean4_qg_hcx'), td('Qwen3.5 · Qwen3.8 · Gemma 4 · Gemma 4 +dev · HCX(×2) 가중 평균'), td('97.7', { align: 'right' }), td('0.97646', { align: 'right' }), td('3,278', { align: 'right' }), td('')],
    [td('Flash-Next 단독'), td('학습 모델 1개 (후보 B의 기준 모델)'), td('96.9', { align: 'right' }), td('0.97110', { align: 'right' }), td('3,260', { align: 'right' }), td('')],
  ];
  s.addTable(rows, tableOpts({ x: M, y: 1.8, w: CW, colW: [2.0, 5.55, 0.95, 1.2, 1.2, CW - 10.9], fontSize: 12.5, rowH: 0.52 }));
  const y = 5.65;
  const notes = [
    ['공개 문항 수 = 점수 × 3,357', '상위 제출끼리는 2~3문항 차이 — 공개 절반에서 약 20문항이 갈려, 운만으로 ±4~5문항은 흔합니다.'],
    ['두 번째 칸은 방식이 다른 후보로', '최종 점수는 두 제출 중 높은 쪽입니다. 검증으로 고른 조합에, 방식이 다른 교체 규칙 파일을 더했습니다.'],
  ];
  const nw = (CW - 0.3) / 2;
  notes.forEach(([h, b], i) => {
    const x = M + i * (nw + 0.3);
    tile(s, x, y + 0.08, 0.36, '!', { fontSize: 13 });
    T(s, [{ text: h, options: { bold: true, color: C.navy, breakLine: true } }, { text: b }], { x: x + 0.52, y, w: nw - 0.55, h: 1.1, fontSize: 12.5, color: C.gray, lineSpacingMultiple: 1.1 });
  });
  footer(s, 14);
  s.addNotes('최종 제출은 검증으로 고른 조합 ea0e9f(공개 0.97736)와 교체 규칙 파일 505d49(공개 0.97378)입니다. 공개 점수는 test의 절반인 3,357문항으로 매겨져 상위 제출끼리 2~3문항 차이는 우연 범위라, 확인용으로만 썼습니다.');
}

// =========================================================
// 15. 채택하지 않은 실험
// =========================================================
{
  const s = pres.addSlide();
  title(s, '채택하지 않은 실험 — 단독 개선 ≠ 조합 개선', '미리 정한 기준으로 판정 — 검증 순이득이 0 이하였거나 공개 점수가 떨어졌습니다.');
  const no = { color: C.gray };
  const rows = [
    [th('실험'), th('결과'), th('판단')],
    [td('학습 변형 19가지 (LR · epoch · LoRA 세기 · 보조 손실)', { bold: true }), td('단독 96.6~97.6%, 앙상블 순이득 −2~0'), td('불채택', no)],
    [td('bf16 재학습 (NF4 대비)', { bold: true }), td('같은 시드면 답 99% 동일 (Qwen3.5 97.2 → 97.1)'), td('병목 아님', no)],
    [td('학습 데이터 3,500문항', { bold: true }), td('5,714문항과 같은 97.2%'), td('데이터 양 충분', no)],
    [td('OCR 글자 프롬프트', { bold: true }), td('Gemma 97.5 → 97.7 · Qwen3.5 97.1 → 96.9'), td('일관된 이득 없음', no)],
    [td('보기 순열 TTA', { bold: true }), td('공개 0.97557 · 0.97587 (원래보다 2~4문항 낮음)'), td('불채택', no)],
    [td('애매한 문항 고해상도 재채점 (3·4배)', { bold: true }), td('멤버 단위 이득 없음'), td('불채택', no)],
    [td('보기별 예/아니오 검증', { bold: true }), td('최고 조합 위에서 98.0 → 97.7'), td('불채택', no)],
    [td('검증 포함 재학습(_full)으로 교체', { bold: true }), td('공개 4문항 하락'), td('불채택', no)],
    [td('dev 의사 라벨 추가 학습', { bold: true }), td('Gemma 97.6 → 97.4 · Qwen3.8 96.8 → 96.3'), td('불채택', no)],
  ];
  s.addTable(rows, tableOpts({ x: M, y: 1.8, w: CW, colW: [4.6, 5.3, CW - 9.9], fontSize: 12.5, rowH: 0.49 }));
  footer(s, 15);
  s.addNotes('단독으로는 좋아 보인 변형이 많았지만, 조합에 넣으면 순이득이 0 이하였습니다. 멤버들이 같은 문항을 확신하며 틀리는 구조라, 한 멤버의 학습 방식 차이는 조합에서 거의 드러나지 않았습니다.');
}

// =========================================================
// 16. 오답 분석
// =========================================================
{
  const s = pres.addSlide();
  title(s, '오답 분석 — 천장은 라벨이 정한다', '검증 기반 조합의 오답 24개를 이미지로 직접 확인하니, 모델로 고칠 수 있는 건 6개 남짓이었습니다.');
  s.addChart(pres.charts.DOUGHNUT, [{ name: '검증 오답 24개', labels: ['라벨·문제 오류', '답이 둘 다 됨', '모델 실수'], values: [10, 8, 6] }], {
    x: M + 0.2, y: 1.95, w: 4.8, h: 4.8, holeSize: 58, chartColors: [C.navy, 'B6BDCA', C.amber],
    showValue: false, showPercent: false, showLabel: false, showLegend: false, showTitle: false,
  });
  T(s, [{ text: '24', options: { fontSize: 36, bold: true, color: C.navy, breakLine: true } }, { text: '검증 오답', options: { fontSize: 12, color: C.gray } }],
    { x: M + 1.6, y: 3.8, w: 2.0, h: 1.1, align: 'center', valign: 'middle' });
  const rx = 6.2, rw = W - M - rx;
  const groups = [
    [C.navy, C.white, '라벨·문제 오류 10', '간판은 ROCK인데 정답 POP · 브랜드 Johnnie Walker인데 정답 Blue Label · 이미지에 없는 TANQUERAY'],
    ['B6BDCA', C.navy, '답이 둘 다 됨 8', '札幌 = Sapporo · 48만 / 40만 · “밍기적” vs “망기적”'],
    [C.amber, C.navy, '모델 실수 6', '거꾸로 붙은 “냉장보관” 스티커 · 작은 글씨 100g · 출구 안내판 배치'],
  ];
  groups.forEach(([fill, tc, h, b], i) => {
    const y = 1.85 + i * 1.45;
    box(s, rx, y, rw, 1.28, { fill: C.light });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: rx + 0.25, y: y + 0.28, w: 0.32, h: 0.32, rectRadius: 0.06, fill: { color: fill }, line: { color: fill } });
    T(s, h, { x: rx + 0.72, y: y + 0.2, w: rw - 1.0, h: 0.45, fontSize: 15, bold: true, valign: 'middle' });
    T(s, b, { x: rx + 0.72, y: y + 0.66, w: rw - 0.95, h: 0.55, fontSize: 12.5, color: C.gray });
  });
  T(s, '학습 라벨은 원본 그대로 두었습니다 (test에도 같은 성격의 문항이 있을 수 있음). 조합으로 고칠 여지는 멤버 의견이 갈리는 문항 몇 개뿐이었습니다.',
    { x: rx, y: 6.2, w: rw, h: 0.7, fontSize: 12, color: C.navy, lineSpacingMultiple: 1.1 });
  footer(s, 16);
  s.addNotes('오답 24개 중 10개는 라벨이나 문제 자체의 오류, 8개는 두 보기가 모두 답이 될 수 있는 문항이었습니다. 멤버 전부가 같은 오답을 확신하는 문항이 많아, 단독 성능을 올려도 이 문항들은 움직이지 않았습니다.');
}

// =========================================================
// 17. 코드 구성과 실행 방법
// =========================================================
{
  const s = pres.addSlide();
  title(s, '코드 구성과 실행 방법', '노트북 번호 순서대로 실행 — 모든 결과는 vqa_runs/<실행>/에 예측 CSV와 train_log.json으로 남습니다.');
  const n = (t) => td(t, { bold: true, fill: { color: C.amber }, align: 'center', color: C.navy });
  const rows = [
    [th('순서', { align: 'center' }), th('노트북'), th('환경'), th('하는 일')],
    [n('0'), td('00_data_prep · 00b_ocr_paddle', { bold: true }), td('CPU · GPU'), td('데이터 점검, 고정 분할 확인, 보조 학습 데이터, OCR')],
    [n('1'), td('01_train_vlm', { bold: true }), td('Colab A100 80GB · G4 96GB'), td('학습 모델 전부 (RECIPE = 실행 이름)')],
    [n('2'), td('02_runpod_flashnext_bf16', { bold: true }), td('RunPod B300'), td('Qwen3.8-Flash-Next 학습')],
    [n('3'), td('03a_zs122b_vllm_colab · 03b_runpod_zs_any', { bold: true }), td('G4 · B300'), td('학습 없는 대형 모델 채점 (vLLM)')],
    [n('4'), td('04_runpod_xcheck_any', { bold: true }), td('B300'), td('판정 모델 (답이 갈린 문항만)')],
    [n('5'), td('05_ens_search · 06_anchor_rescue', { bold: true }), td('CPU'), td('조합 · 교체 규칙 → 제출 CSV + 설정 JSON')],
    [n('6'), td('07_final_check', { bold: true }), td('CPU'), td('고른 CSV 형식 점검 + 설정 JSON만으로 다시 만들어 대조')],
    [n('—'), td('util_runpod_bridge_colab', { bold: true }), td('CPU'), td('Drive ↔ HF 비공개 저장소 ↔ RunPod 파일 옮기기')],
  ];
  s.addTable(rows, tableOpts({ x: M, y: 1.8, w: CW, colW: [0.8, 4.5, 2.6, CW - 7.9], fontSize: 12.5, rowH: 0.47 }));
  T(s, [
    { text: 'README.md', options: { bold: true, color: C.navy } }, { text: ' 실행 순서 · 환경 · 재현 방법   ' },
    { text: 'MODEL_NOTES.md', options: { bold: true, color: C.navy } }, { text: ' 모델별 처리(HCX 답 자리 고정, 해상도, Flash-Next 보정, vLLM 설정)' },
  ], { x: M, y: 6.6, w: CW, h: 0.4, fontSize: 12.5, color: C.gray });
  footer(s, 17);
  s.addNotes('제출 CSV만 다시 만들 때는 저장된 예측으로 5·6번 노트북만 돌리면 됩니다. RunPod에는 Google Drive가 붙지 않아 HF 비공개 데이터셋 저장소로 데이터와 결과를 옮겼습니다.');
}

// =========================================================
// 18. 재현성
// =========================================================
{
  const s = pres.addSlide();
  title(s, '재현성 — 세 단계로 확인', '제출 CSV는 저장된 예측과 설정 JSON만으로 난수 없이 다시 만들어집니다.');
  const steps = [
    ['저장된 예측 → 제출 CSV', '07_final_check가 독립 코드로 다시 계산하고 멤버 예측 파일의 md5를 대조', '완전히 같음'],
    ['저장된 어댑터 → 예측', '01_train_vlm에서 REUSE_FINISHED = True로 검증 일부를 새로 채점해 저장된 예측과 비교', '같은 GPU면 답 ~100% 일치'],
    ['처음부터 재학습', 'RUN_SUFFIX = "_repro"로 같은 레시피를 다시 학습 (시드·순서·설정 동일)', 'GPU 비결정성으로 소수 문항 차이 가능'],
  ];
  const sw = (CW - 2 * 0.45) / 3;
  steps.forEach(([h, b, exp], i) => {
    const x = M + i * (sw + 0.45), y = 1.85;
    box(s, x, y, sw, 2.85, { fill: C.light });
    tile(s, x + 0.25, y + 0.25, 0.5, String(i + 1), { fontSize: 17 });
    T(s, h, { x: x + 0.9, y: y + 0.25, w: sw - 1.1, h: 0.5, fontSize: 15.5, bold: true, valign: 'middle' });
    T(s, b, { x: x + 0.25, y: y + 0.95, w: sw - 0.5, h: 1.1, fontSize: 12.5, color: C.gray, lineSpacingMultiple: 1.1 });
    s.addText(exp, { shape: pres.shapes.ROUNDED_RECTANGLE, x: x + 0.25, y: y + 2.1, w: sw - 0.5, h: 0.52, rectRadius: 0.08,
      fill: { color: i === 0 ? C.amber : C.white }, line: { color: i === 0 ? C.amber : C.line }, fontFace: F, fontSize: 12.5, bold: true, color: C.navy, align: 'center', valign: 'middle', margin: 0, isTextBox: true });
    if (i < 2) arrow(s, x + sw + 0.07, y + 1.42, x + sw + 0.38, y + 1.42, C.navy);
  });
  const bw = (CW - 0.3) / 2, by = 5.0;
  box(s, M, by, bw, 1.7, { fill: C.white, line: C.line });
  T(s, [
    { text: '고정한 것', options: { bold: true, color: C.navy, fontSize: 14, breakLine: true } },
    { text: '검증 분할(시드 42 · md5) · 학습 시드(문항 순서 · 보기 섞기 · LoRA 초기값) · 판정 모델 문항별 시드 · transformers 5.17.0(HCX만 개발 커밋) · peft 0.21.0 · bitsandbytes 0.50.2 · vLLM 0.30.0' },
  ], { x: M + 0.25, y: by + 0.16, w: bw - 0.5, h: 1.45, fontSize: 12.5, color: C.gray, lineSpacingMultiple: 1.1 });
  box(s, M + bw + 0.3, by, bw, 1.7, { fill: C.white, line: C.line });
  T(s, [
    { text: '코드 점검', options: { bold: true, color: C.navy, fontSize: 14, breakLine: true } },
    { text: '작은 모델(CPU)로 학습 노트북 9개 경로를 끝까지 실행: 점검 실행 · 본 학습 · 비전 LoRA · 이어학습+보조 · OCR · 기록 대조 · 전체 데이터 · dev 의사 라벨 · 중단 후 이어하기(한 번에 학습한 결과와 동일)' },
  ], { x: M + bw + 0.55, y: by + 0.16, w: bw - 0.5, h: 1.45, fontSize: 12.5, color: C.gray, lineSpacingMultiple: 1.1 });
  footer(s, 18);
  s.addNotes('1단계는 난수가 없어 완전히 같은 CSV가 나와야 합니다. 2단계는 저장된 어댑터로 다시 채점해 예측을 대조하고, 3단계는 처음부터 다시 학습하는 경우로 GPU 커널의 비결정성 때문에 소수 문항이 다를 수 있습니다.');
}

// =========================================================
// 19. 한계와 배운 점 (마무리)
// =========================================================
{
  const s = pres.addSlide();
  s.background = { color: C.navy };
  T(s, '한계와 배운 점', { x: M, y: 0.42, w: CW, h: 0.72, fontSize: 30, bold: true, color: C.white, valign: 'middle' });
  T(s, '모델을 바꾸는 것보다, 무엇이 진짜 개선인지 가려내는 절차가 더 중요했습니다.', { x: M, y: 1.14, w: CW, h: 0.42, fontSize: 15, color: C.ice });
  const items = [
    ['천장은 라벨이 정한다', '검증 오답의 절반 이상이 라벨 오류·복수 정답이고, 멤버 전부가 같은 오답을 확신했습니다.'],
    ['검증 1,000문항은 작았다', '±3문항은 우연과 구별하기 어렵습니다. 학습은 3,500문항이면 충분해, 검증을 더 크게 둘 수 있었습니다.'],
    ['같은 문항 비교 · 사전 규칙', '고친·새로 틀린 문항으로 판정하고 기준을 먼저 정하니, 단독으로만 좋아 보이는 변형을 걸러낼 수 있었습니다.'],
    ['최고값은 부풀려진다', '조합 수십만 개 중 1등은 중첩 CV에서 약 0.5%p 낮았고, 공개 최고를 고를 때도 같습니다.'],
    ['다양성이 이득을 만든다', '조합에 들어간 건 단독 순위보다 틀리는 문항이 다른 모델(InternVL · OCR 모델)이었습니다.'],
    ['정밀도는 병목이 아니었다', '같은 시드의 NF4·bf16 답이 99% 같았고, 초대형 모델은 사전 양자화 체크포인트로 썼습니다.'],
  ];
  const w3 = (CW - 2 * 0.3) / 3, h3 = 2.3;
  items.forEach(([h, b], i) => {
    const x = M + (i % 3) * (w3 + 0.3), y = 1.85 + Math.floor(i / 3) * (h3 + 0.3);
    infoCard(s, x, y, w3, h3, h, b, { num: i + 1, fill: C.navy2, headColor: C.white, bodyColor: C.ice, bodySize: 14, headSize: 16 });
  });
  T(s, TEAM, { x: M, y: 6.95, w: 8, h: 0.35, fontSize: 13, bold: true, color: C.amber });
  T(s, '19', { x: W - M - 1, y: 7.06, w: 1, h: 0.26, fontSize: 9, color: '8793AD', align: 'right' });
  s.addNotes('가장 큰 제약은 라벨 노이즈와 작은 검증셋이었습니다. 그래서 모델을 더 만드는 것보다, 같은 문항 비교와 미리 정한 규칙, 중첩 교차검증으로 진짜 개선만 골라내는 절차에 힘을 썼습니다.');
}

pres.writeFile({ fileName: path.join(__dirname, 'A084_VQA_solution.pptx') }).then((f) => console.log('saved', f));
