### Data Issues

- Cap opinions with undersplitting
- Cap opinions with null authors
- Missing docket numbers from super scotus means missing cases in prediction set
- Incorrect opinions pulled from CAP (longest casebody not guaranteed to return opinion)
    - 87-1224 opinion is only around 1 sentence, so we pulled a longer consolidation filing related to the case instead
        - Correct opinion: https://cite.case.law/us/488/590/
        - Incorrect opinion that made it into dataset: https://cite.case.law/us/489/1092/12090236/
    
- There are about 50 CAP opinions SCDB says shouldn't exist, but they do
    - Use CAP as starting point for cases and make sure we have at least as many as we expect from SCDB

#### Observed causes of null author opinions
- Unknown cause (e.g https://api.case.law/v1/cases/1481163/?full_case=true)
- Opinion starts with finding as in https://api.case.law/v1/cases/12581203/?full_case=true 
    >"Held : The judgment is reversed.\n723 F.3d 434, reversed.\nJustice KENNEDY de"
- Missing big chunk of opinion
    - Compare Thomas' plurality opinion https://api.case.law/v1/cases/12616942/?full_case=true with https://supreme.justia.com/cases/federal/us/583/16-498/#tab-opinion-3858471
    - This case is also interesting because it's a 6-3 decision with no ootc, just a 3-justice plurality opinion by Thomas
- Incorrect case grabbed based on opinion length heuristic
    - See https://old.case.law/search/#/?page=1&ordering=relevance&docket_number=91-6646
        - expected https://cite.case.law/us/506/19/
        - got https://cite.case.law/us/506/1087/12103909/
- Oversplitting as in https://api.case.law/v1/cases/12455275/?full_case=true with null-author opinion with complete text of:
    > SEPARATE OPINION
    - This is an issue of using the Lawyer's Edition reporter. US Reports version of this case doesn't have this.
- Empty opinion as in https://api.case.law/v1/cases/12623718/?full_case=true
    - Here this was an issue of picking the wrong opinion due to length heuristic
    - got https://cite.case.law/s-ct/139/626/12623718/
    - expected https://cite.case.law/s-ct/140/2412/12626632/
