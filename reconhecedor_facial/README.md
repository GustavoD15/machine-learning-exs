# Reconhecimento facial

Solução em script Python usando **OpenCV** para leitura, desenho e gravação
de imagens e **DeepFace** para detecção e comparação de rostos com Facenet512.

## Estrutura

```text
reconhecedor_facial/
├── reconhecimento.py
├── requirements.txt
├── autorizados/
│   └── pessoa_exemplo/     # Substitua pelo nome da pessoa e adicione suas fotos
├── testes/                # Fotos a serem avaliadas
└── resultados/            # Imagens anotadas geradas pelo script
```

## Instalação e execução (PowerShell)

Use Python 3.10, 3.11 ou 3.12. A instalação local foi preparada com Python 3.10.
Dentro desta pasta (`reconhecedor_facial`):

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe reconhecimento.py
```

Antes de executar:

1. Crie uma subpasta em [autorizados](autorizados/) para cada pessoa, por exemplo
   `autorizados/ana/`, e adicione uma ou mais fotos contendo **somente essa pessoa**.
2. Coloque fotos diferentes das fotos de cadastro em [testes](testes/), incluindo
   pessoas cadastradas, não cadastradas e uma foto com vários rostos.
3. Execute [reconhecimento.py](reconhecimento.py) e confira [resultados](resultados/).

São aceitos JPG, JPEG, PNG, BMP e WEBP, inclusive em subpastas. Fotos de cadastro
com nenhum ou vários rostos são descartadas. Os diretórios são inicialmente vazios:
as fotos devem ser fornecidas por você. As imagens em `autorizados/`, `testes/`
e `resultados/` podem ser versionadas no Git.

Cada rosto recebe sua própria **bounding box**:

- **Verde**, com o texto exato `acesso liberado`, quando corresponde a um cadastro.
- **Vermelha**, com o texto exato `acesso negado`, quando não corresponde.

Sem cadastros válidos, todo rosto é negado. Se nenhum rosto for detectado,
a imagem inteira recebe uma caixa vermelha. Imagens ilegíveis são informadas
no terminal e ignoradas. Os originais são preservados; novas execuções substituem
os resultados com o mesmo nome.

Os caminhos padrão são relativos ao script. Para usar outras pastas:

```powershell
.\.venv\Scripts\python.exe reconhecimento.py --autorizados "C:\fotos\cadastros" --testes "C:\fotos\avaliacao" --saida "C:\fotos\resultados"
```

A primeira comparação pode baixar os pesos do modelo pela internet, armazenando-os
na pasta `.deepface` dentro deste projeto (ignorada pelo Git). A variável
`DEEPFACE_HOME`, quando definida, permite usar outro local. A instalação inclui TensorFlow por meio
das dependências e pode ocupar bastante espaço.

## Validação manual

Confira uma foto nova de uma pessoa cadastrada (verde), uma pessoa não cadastrada
(vermelho), ambas na mesma foto (uma caixa por rosto) e uma imagem sem pessoas
(vermelho na imagem inteira). A qualidade, iluminação e posição dos rostos afetam
o resultado; a comparação usa o limiar padrão do DeepFace para Facenet512/cosseno.
