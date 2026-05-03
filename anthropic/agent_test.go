package anthropic

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"strings"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/inancsege/fugue"
)

// fakeTransport returns canned responses in order. Captures requests for
// assertion. Each test constructs one inline with the responses it needs.
type fakeTransport struct {
	responses     []*http.Response
	requests      []*http.Request
	requestBodies [][]byte // captured before the body is consumed
	cursor        int

	// closeCalls tracks how many response bodies have been Close()d. Used to
	// verify Stream releases the connection on early termination.
	closeCalls int
}

func (t *fakeTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if req.Body != nil {
		body, err := io.ReadAll(req.Body)
		if err != nil {
			return nil, fmt.Errorf("fakeTransport: read body: %w", err)
		}
		req.Body.Close()
		t.requestBodies = append(t.requestBodies, body)
		req.Body = io.NopCloser(bytes.NewReader(body))
	} else {
		t.requestBodies = append(t.requestBodies, nil)
	}
	t.requests = append(t.requests, req)
	if t.cursor >= len(t.responses) {
		return nil, fmt.Errorf("fakeTransport: out of canned responses (cursor=%d)", t.cursor)
	}
	r := t.responses[t.cursor]
	t.cursor++
	if r.Body != nil {
		r.Body = &countingCloser{ReadCloser: r.Body, parent: t}
	}
	return r, nil
}

type countingCloser struct {
	io.ReadCloser
	parent *fakeTransport
	closed bool
}

func (c *countingCloser) Close() error {
	if !c.closed {
		c.closed = true
		c.parent.closeCalls++
	}
	return c.ReadCloser.Close()
}

// okResponse builds a 200 application/json response from a body string.
func okResponse(body string) *http.Response {
	return &http.Response{
		StatusCode: 200,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(strings.NewReader(body)),
	}
}

// sseResponse builds a 200 text/event-stream response from one or more SSE
// event blocks. Each event must include the trailing blank line.
func sseResponse(events ...string) *http.Response {
	return &http.Response{
		StatusCode: 200,
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
		Body:       io.NopCloser(strings.NewReader(strings.Join(events, ""))),
	}
}

// newAgentWithTransport wires fakeTransport into an agent via WithClient.
func newAgentWithTransport(model string, ft *fakeTransport, opts ...Option) fugue.Agent {
	httpClient := &http.Client{Transport: ft}
	c := sdk.NewClient(option.WithHTTPClient(httpClient), option.WithAPIKey("test-key"))
	all := append([]Option{WithClient(&c)}, opts...)
	return New(model, all...)
}

// msg is a tiny helper for building text-only Messages in tests.
func msg(role fugue.Role, text string) fugue.Message {
	return fugue.Message{Role: role, Content: []fugue.Part{fugue.Text{Text: text}}}
}
